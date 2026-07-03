"""
Tree-BIC Score for Bayesian Network Structure Learning
=======================================================
Replaces the standard flat CPT parametrisation in BIC with an MDL-optimal
decision tree (Quinlan & Rivest, 1989) that represents P(X | Pa(X)).

Key ideas
---------
Standard BIC for a node X with parents Pa(X):

    BIC(X, Pa) = log-likelihood  -  (k/2) * log(N)

where k = number of free parameters in the flat CPT =
    (|X| - 1) * prod_{Y in Pa} |Y|.

Here we replace the flat CPT with an MDL decision tree over Pa(X).
The tree may share/merge contexts, so its effective parameter count is:

    k_tree = (|X| - 1) * (number of leaves in the trimmed tree)

and the log-likelihood term is computed using the leaf-level empirical
distributions rather than the full joint.

This gives a Tree-BIC score:

    TreeBIC(X, Pa) = LL_tree(X, Pa)  -  (k_tree / 2) * log(N)

where LL_tree = sum over leaves L of  sum_{x} N(x|L) * log( N(x|L)/N(L) )

pgmpy compatibility
-------------------
pgmpy's StructureScore interface requires a class with:

    local_score(variable: str, parents: list[str]) -> float

and optionally:

    score(model: BayesianNetwork) -> float  (sum of local scores)

This module provides `TreeBICScore` that satisfies both requirements.
It can be passed directly to pgmpy structure-learning algorithms:

    from pgmpy.estimators import HillClimbSearch
    from tree_bic_score import TreeBICScore

    scorer = TreeBICScore(data)
    hc = HillClimbSearch(data)
    best_model = hc.estimate(scoring_method=scorer)

Standalone usage (without pgmpy)
---------------------------------
    scorer = TreeBICScore(data)          # data: pd.DataFrame or list[dict]
    scorer.local_score("X", ["A", "B"]) # -> float
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from typing import Any, Optional

import sys

# ---------------------------------------------------------------------------
# Optional pgmpy import — gracefully degrade if not installed
# ---------------------------------------------------------------------------
try:
    from pgmpy.estimators import StructureScore          # pgmpy ≥ 0.1.17
    _PGMPY_BASE = StructureScore
except ImportError:
    # Provide a minimal stand-in so the class definition still works
    class _StructureScoreStub:                           # type: ignore[misc]
        """Minimal stub replicating pgmpy.estimators.StructureScore."""
        def __init__(self, data, **kwargs):
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                self.data = data
            else:
                import pandas as pd
                self.data = pd.DataFrame(data)
            self.state_names: dict[str, list] = {
                col: sorted(self.data[col].unique().tolist())
                for col in self.data.columns
            }

        def score(self, model) -> float:
            """Sum local scores over all nodes in the model."""
            return sum(
                self.local_score(node, list(model.predecessors(node)))
                for node in model.nodes()
            )

    _PGMPY_BASE = _StructureScoreStub


# ============================================================
# MDL Decision Tree primitives (self-contained copy)
# ============================================================

def _entropy(labels: list) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n)
                for c in Counter(labels).values() if c > 0)


def _log_star(n: int) -> float:
    """Simplified universal integer code length."""
    return math.log2(n + 1) if n > 0 else 0.0


def _dl_leaf(labels: list, num_classes: int) -> float:
    """MDL description length of a leaf node (in bits)."""
    return _log_star(num_classes) + _entropy(labels) * len(labels)


def _dl_node(node: dict, num_classes: int) -> float:
    if node["type"] == "leaf":
        return _dl_leaf(node["labels"], num_classes)
    
    cost = _log_star(node["num_features"])
    cost += sum(_dl_node(c, num_classes) for c in node["children"].values())

    #cost = _log_star(node["num_features"])
    #n_children = len(node["children"])
    #cost += math.log2(n_children + 1)                   # encode arity
    #cost += math.log2(n_children) * n_children          # encode which value -> which child (approx)
    #cost += sum(_dl_node(c, num_classes) for c in node["children"].values())

    return cost


def _make_leaf(labels: list) -> dict:
    return {"type": "leaf", "labels": labels,
            "prediction": Counter(labels).most_common(1)[0][0] if labels else None}


def _make_internal(feature, num_features, children, labels) -> dict:
    return {"type": "internal", "feature": feature,
            "num_features": num_features,
            "children": children, "labels": labels}


def _grow(labels, data, features, used, all_classes, num_features) -> dict:
    num_classes = len(all_classes)
    if not labels or len(set(labels)) == 1 or used == frozenset(features):
        return _make_leaf(labels)

    best_feat, best_score = None, math.inf
    for feat in features:
        if feat in used:
            continue
        groups: dict = {}
        for row, lbl in zip(data, labels):
            groups.setdefault(row[feat], []).append(lbl)
       
        split_dl = (_log_star(num_features) + sum(_dl_leaf(g, num_classes) for g in groups.values()))
        if split_dl < best_score:
            best_score, best_feat = split_dl, feat

    if best_feat is None or best_score >= _dl_leaf(labels, num_classes):
        return _make_leaf(labels)

    groups_d: dict = {}
    for row, lbl in zip(data, labels):
        v = row[best_feat]
        groups_d.setdefault(v, ([], []))
        groups_d[v][0].append(lbl)
        groups_d[v][1].append(row)

    new_used = used | {best_feat}
    children = {
        v: _grow(lbls, rows, features, new_used, all_classes, num_features)
        for v, (lbls, rows) in groups_d.items()
    }
    return _make_internal(best_feat, num_features, children, labels)


def _trim(node: dict, num_classes: int) -> dict:
    if node["type"] == "leaf":
        return node
    trimmed = {v: _trim(c, num_classes) for v, c in node["children"].items()}
    node = {**node, "children": trimmed}
    if _dl_node(node, num_classes) >= _dl_leaf(node["labels"], num_classes):
        return _make_leaf(node["labels"])
    return node


def _count_leaves(node: dict) -> int:
    if node["type"] == "leaf":
        return 1
    return sum(_count_leaves(c) for c in node["children"].values())


def _leaf_label_counts(node: dict) -> list[Counter]:
    """Return a Counter of label frequencies for every leaf."""
    if node["type"] == "leaf":
        return [Counter(node["labels"])]
    result = []
    for c in node["children"].values():
        result.extend(_leaf_label_counts(c))
    return result


# ============================================================
# Tree-BIC Score
# ============================================================

class TreeBICScore(_PGMPY_BASE):
    r"""
    Tree-BIC structure score for Bayesian Network learning.

    Replaces flat CPTs with MDL-optimal decision trees (Quinlan & Rivest 1989).

    Score for a single node X given parents Pa:

        TreeBIC(X, Pa) = LL_tree  -  (k_tree / 2) * log(N)

    where
        LL_tree  = Σ_{leaves l} Σ_x  N(x, l) · log[ N(x, l) / N(l) ]
                   (log-likelihood under tree-structured P(X | Pa))
        k_tree   = (|X| - 1) · L      (L = number of leaves after trimming)
        N        = total sample size

    When Pa = ∅ the tree degenerates to a single leaf and the score
    reduces to the standard BIC marginal score for X.

    Parameters
    ----------
    data : pd.DataFrame or list[dict]
        Fully observed, discrete data.
    state_names : dict[str, list], optional
        Maps each variable to its ordered list of states.
        Inferred from data if not provided.
    equivalent_sample_size : float
        Not used in BIC (here for API parity with BDeu).
    """

    def __init__(self, data, state_names: Optional[dict] = None, **kwargs):
        # ---- normalise data to pandas ----
        try:
            import pandas as pd
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
        except ImportError:
            raise ImportError("pandas is required for TreeBICScore.")

        # Call parent __init__ (pgmpy) or stub __init__
        super().__init__(data, **kwargs)

        # state_names: prefer explicit argument, then parent, then infer
        if state_names is not None:
            self.state_names = state_names
        elif not hasattr(self, "state_names") or self.state_names is None:
            self.state_names = {
                col: sorted(data[col].dropna().unique().tolist())
                for col in data.columns
            }

        self._data_records: list[dict] = data.to_dict("records")
        self._N = len(data)

    # ------------------------------------------------------------------
    # Core public method — pgmpy interface
    # ------------------------------------------------------------------

    def local_score(self, variable: str, parents: list[str]) -> float:
        """
        Compute TreeBIC(variable | parents).

        Parameters
        ----------
        variable : str
            The child node X.
        parents : list[str]
            Parent nodes Pa(X).  May be empty.

        Returns
        -------
        float
            Tree-BIC score (higher is better).
        """
        if self._N == 0:
            return 0.0

        var_states = self.state_names[variable]
        num_classes = len(var_states)
        parents = list(parents)

        labels = [r[variable] for r in self._data_records]

        if not parents:
            # No parents — degenerate tree = single leaf
            ll = self._leaf_ll(Counter(labels))
            k = num_classes - 1
            return ll - 0.5 * k * math.log(self._N)

        # ---- Build MDL decision tree ----
        data_rows = self._data_records          # full rows, tree only reads parents
        num_features = len(parents)
        all_classes = var_states

        root = _grow(
            labels=labels,
            data=data_rows,
            features=parents,
            used=frozenset(),
            all_classes=all_classes,
            num_features=num_features,
        )
        root = _trim(root, num_classes)

        # ---- Compute log-likelihood under tree ----
        leaf_counters = _leaf_label_counts(root)
        ll = sum(self._leaf_ll(cnt) for cnt in leaf_counters)

        # ---- Effective parameter count ----
        n_leaves = _count_leaves(root)
        k = (num_classes - 1) * n_leaves

        return ll - 0.5 * k * math.log(self._N)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _leaf_ll(counts: Counter) -> float:
        """
        Log-likelihood contribution of one leaf:
            Σ_x  N_x · log(N_x / N_leaf)
        """
        n_leaf = sum(counts.values())
        if n_leaf == 0:
            return 0.0
        return sum(
            nx * math.log(nx / n_leaf)
            for nx in counts.values()
            if nx > 0
        )

    def score(self, model) -> float:
        """
        Total TreeBIC score of a BayesianNetwork (or any nx.DiGraph).

        Parameters
        ----------
        model : pgmpy.models.BayesianNetwork or nx.DiGraph

        Returns
        -------
        float
            Sum of local scores.
        """
        return sum(
            self.local_score(node, list(model.predecessors(node)))
            for node in model.nodes()
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def describe_local(self, variable: str, parents: list[str]) -> str:
        """
        Return a human-readable breakdown of the TreeBIC score,
        including the learned decision tree.
        """
        parents = list(parents)
        var_states = self.state_names[variable]
        num_classes = len(var_states)
        labels = [r[variable] for r in self._data_records]

        lines = [
            f"TreeBIC  {variable}  |  {parents}",
            f"  N = {self._N},  |{variable}| = {num_classes}",
        ]

        if not parents:
            cnt = Counter(labels)
            ll = self._leaf_ll(cnt)
            k = num_classes - 1
            sc = ll - 0.5 * k * math.log(self._N)
            lines += [
                "  (no parents — single leaf)",
                f"  LL = {ll:.3f},  k = {k},  penalty = {0.5*k*math.log(self._N):.3f}",
                f"  TreeBIC = {sc:.3f}",
            ]
            return "\n".join(lines)

        root = _grow(labels, self._data_records, parents,
                     frozenset(), var_states, len(parents))
        root = _trim(root, num_classes)

        leaf_counters = _leaf_label_counts(root)
        ll = sum(self._leaf_ll(cnt) for cnt in leaf_counters)
        n_leaves = _count_leaves(root)
        k = (num_classes - 1) * n_leaves
        penalty = 0.5 * k * math.log(self._N)
        sc = ll - penalty

        lines += [
            f"  Leaves = {n_leaves},  k_tree = {k}",
            f"  LL_tree = {ll:.3f},  penalty = {penalty:.3f}",
            f"  TreeBIC = {sc:.3f}",
            "",
            "  Decision tree:",
        ]
        lines += _format_tree(root, indent=4)

        return "\n".join(lines)


# ============================================================
# Tree pretty-printer
# ============================================================

def _format_tree(node: dict, indent: int = 0, prefix: str = "") -> list[str]:
    pad = " " * indent
    if node["type"] == "leaf":
        cnt = Counter(node["labels"])
        dist = ", ".join(f"{k}:{v}" for k, v in sorted(cnt.items()))
        return [f"{pad}{prefix}LEAF [{dist}]  (n={len(node['labels'])})"]
    lines = [f"{pad}{prefix}SPLIT '{node['feature']}'"]
    for val, child in sorted(node["children"].items(), key=lambda x: str(x[0])):
        lines += _format_tree(child, indent + 4, prefix=f"[{val!r}] → ")
    return lines


# ============================================================
# Self-contained demo (no pgmpy required)
# ============================================================

if __name__ == "__main__":
    import pandas as pd
    import random

    random.seed(0)

    # ------------------------------------------------------------------ #
    # Synthetic dataset: X depends on A and B through a tree structure    #
    # ------------------------------------------------------------------ #
    N = 500
    rows = []
    for _ in range(N):
        a = random.choice(["a1", "a2", "a3"])
        b = random.choice(["b1", "b2"])
        c = random.choice(["c1", "c2"])          # noise variable

        # Ground-truth: X = 'yes' if a=='a1', 'no' if a=='a3',
        #               else follows b
        if a == "a1":
            x = "yes"
        elif a == "a3":
            x = "no"
        else:
            x = "yes" if b == "b1" else "no"

        rows.append({"A": a, "B": b, "C": c, "X": x})

    df = pd.DataFrame(rows)

    scorer = TreeBICScore(df)

    print("=" * 60)
    print("Tree-BIC Score Demo")
    print("=" * 60)

    # --- local scores for X ---
    configs = [
        ("X", []),
        ("X", ["A"]),
        ("X", ["B"]),
        ("X", ["C"]),
        ("X", ["A", "B"]),
        ("X", ["A", "B", "C"]),
    ]

    for var, parents in configs:
        sc = scorer.local_score(var, parents)
        print(f"  local_score({var!r}, {parents}) = {sc:.2f}")

    print()
    # Detailed breakdown for the best meaningful config
    print(scorer.describe_local("X", ["A", "B"]))

    print()
    print("=" * 60)
    print("Comparison with flat CPT parameter counts:")
    print("  Pa={A,B}  → flat k =", (2-1) * 3 * 2,
          "  tree k = (|X|-1)*leaves (see above)")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Show pgmpy integration if available                                 #
    # ------------------------------------------------------------------ #
    try:
        from pgmpy.models import BayesianNetwork
        from pgmpy.estimators import HillClimbSearch

        print("\n--- pgmpy HillClimbSearch with TreeBICScore ---")
        hc = HillClimbSearch(df)
        best = hc.estimate(scoring_method=scorer, max_indegree=3, show_progress=False)
        print("Learned edges:", list(best.edges()))
        print("Total TreeBIC:", scorer.score(best))

    except ImportError:
        print("\n(pgmpy not installed — skipping HillClimbSearch demo)")
        print("Install with:  pip install pgmpy")
        print("Then use:")
        print("  from pgmpy.estimators import HillClimbSearch")
        print("  hc = HillClimbSearch(df)")
        print("  model = hc.estimate(scoring_method=TreeBICScore(df))")
