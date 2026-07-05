# -*- coding: utf-8 -*-
"""MIMIC - ALARM.ipynb
"""


import copy
import logging
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "LLM-guided-CBNs"))

import time
import numpy as np
import pandas as pd
import networkx as nx
from itertools import permutations

from pgmpy.utils import get_example_model
from pgmpy.base import DAG
from pgmpy.estimators import TreeSearch, BayesianEstimator, BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from cdt.metrics import SHD as cdt_SHD, SID as cdt_SID

# ---------------------------------------------------------------
# Logging — writes to mimic_alarm.log and stdout
# ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("mimic_alarm.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------
# 1. Load ALARM ground-truth network and dataset
# ---------------------------------------------------------------
DATA_PATH = "domains/alarm/dataset.csv"

log.info("Loading ALARM network...")
true_model = get_example_model("alarm")
variables = list(true_model.nodes())
n = len(variables)
log.info(f"n = {n} variables, {len(true_model.edges())} true edges")

log.info(f"Loading dataset from {DATA_PATH}...")
data = pd.read_csv(DATA_PATH)[variables]  # enforce consistent column order
log.info(f"Dataset: {len(data)} samples, {len(data.columns)} variables")

bic = BicScore(data)
pairs = list(permutations(variables, 2))  # all n(n-1) candidate directed edges
n_edges = len(pairs)
log.info(f"Candidate edge-indicator space size: {n_edges}")

# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------
def repair_acyclic(edge_set, freq=None):
    G = nx.DiGraph()
    G.add_nodes_from(variables)
    G.add_edges_from(edge_set)
    while not nx.is_directed_acyclic_graph(G):
        cycle = nx.find_cycle(G, orientation="original")
        worst = min(
            [(u, v) for u, v, _ in cycle],
            key=lambda e: (freq.get(e, 0) if freq else 0, e[0], e[1]),
        )
        G.remove_edge(*worst)
    return frozenset(G.edges())

def score_dag(edge_set):
    G = DAG()
    G.add_nodes_from(variables)
    G.add_edges_from(edge_set)
    return bic.score(G)

def _to_bn(edge_set):
    G = BayesianNetwork()
    G.add_nodes_from(variables)
    G.add_edges_from(edge_set)
    return G

def compute_metrics(edge_set):
    test = _to_bn(edge_set)
    expert_edges = set(map(tuple, true_model.edges()))
    test_edges   = set(map(tuple, edge_set))
    tp = len(test_edges & expert_edges)
    precision = tp / len(test_edges) if test_edges else 0.0
    recall    = tp / len(expert_edges) if expert_edges else 0.0
    return {
        "SHD":      int(cdt_SHD(copy.deepcopy(true_model), test, double_for_anticausal=False)),
        "SID":      int(cdt_SID(copy.deepcopy(true_model), test)),
        "spurious": len(test_edges - expert_edges),
        "precision": round(precision, 1),
        "recall":    round(recall, 1),
    }

def best_of(pop, scores):
    idx = np.argmax(scores)
    return pop[idx], scores[idx]

def top_k(pop, scores, k):
    ranked = sorted(zip(pop, scores), key=lambda ps: ps[1], reverse=True)
    return [g for g, s in ranked[:k]]

def sample_from_elite(E_df, m, smoothing=.01):
    """
    Fit a Chow-Liu tree over the elite set's varying edge-indicator columns,
    add any constant columns as isolated (parentless) nodes in the same
    network, fit all CPDs with Dirichlet smoothing, and sample m new rows.

    A column unanimous across the elite set still gets its own
    Dirichlet-smoothed marginal CPD -- high probability of staying as-is,
    but never locked in with certainty, since that "unanimous" pattern
    could just be small-elite-set noise rather than a real signal.
    """
    varying_cols = [c for c in E_df.columns if E_df[c].nunique() > 1]

    if len(varying_cols) >= 2:
        tree = TreeSearch(E_df[varying_cols], root_node=varying_cols[0]).estimate(
            estimator_type="chow-liu", show_progress=False
        )
        bn = BayesianNetwork(tree)
    else:
        bn = BayesianNetwork()
        bn.add_nodes_from(varying_cols)

    bn.add_nodes_from(c for c in E_df.columns if c not in bn.nodes())  # constant cols, isolated

    be = BayesianEstimator(bn, E_df)
    for node in bn.nodes():
        bn.add_cpds(be.estimate_cpd(node, prior_type="dirichlet", pseudo_counts=smoothing))

    return BayesianModelSampling(bn).forward_sample(size=m, show_progress=False)[E_df.columns]

# ---------------------------------------------------------------
# 2. MIMIC main loop, with checkpoints at several iteration counts
# ---------------------------------------------------------------
M =  50                  # population size
P_SPARSE = 2.0 / n     # edge probability for random init via DAG.get_random
N_PERCENTILE = 86
CHECKPOINTS = [1, 2, 3, 4, 5]
MAX_ITERS = max(CHECKPOINTS)
SEEDS = [0, 1, 2, 3, 4]
OUT_DIR = "mimic_alarm_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

all_results = []

for seed_val in SEEDS:
    np.random.seed(seed_val)
    log.info(f"=== Seed {seed_val} ===")

    log.info("Initializing population (DAG.get_random)...")
    _idx_to_var = {i: variables[i] for i in range(n)}
    pop = [
        frozenset(
            nx.relabel_nodes(
                DAG.get_random(n_nodes=n, edge_prob=P_SPARSE),
                _idx_to_var,
            ).edges()
        )
        for _ in range(M)
    ]
    scores = [score_dag(g) for g in pop]
    theta = np.percentile(scores, N_PERCENTILE)

    best_dag, best_score = best_of(pop, scores)

    t0 = time.time()
    for it in range(1, MAX_ITERS + 1):
        # ---- select elite ----
        elite = [g for g, s in zip(pop, scores) if s >= theta]
        if len(elite) < 8:
            elite = top_k(pop, scores, 7)

        # ---- fit Chow-Liu tree over edge-indicators, sample new population ----
        E_df = pd.DataFrame(
            [[1 if pr in g else 0 for pr in pairs] for g in elite],
            columns=[f"{a}->{b}" for (a, b) in pairs],
        )

        # edge frequencies across elite set, used for cycle repair
        elite_freq = {pairs[i]: int(E_df.iloc[:, i].sum()) for i in range(len(pairs))}

        samp_df = sample_from_elite(E_df, M)

        new_pop = [
            repair_acyclic(
                {(a, b) for c, (a, b) in zip(E_df.columns, pairs) if samp_df.iloc[k][c] == 1},
                freq=elite_freq,
            )
            for k in range(M)
        ]

        new_scores = [score_dag(g) for g in new_pop]

        candidate_dag, candidate_score = best_of(new_pop, new_scores)
        if candidate_score > best_score:
            best_dag, best_score = candidate_dag, candidate_score

        theta = np.percentile(new_scores, N_PERCENTILE)
        pop, scores = new_pop, new_scores

        if it in CHECKPOINTS:
            m = compute_metrics(best_dag)
            row = {
                "seed": seed_val,
                "iteration": it,
                "best_BIC": round(best_score, 1),
                "SHD": m["SHD"],
                "SID": m["SID"],
                "spurious": m["spurious"],
                "precision": m["precision"],
                "recall": m["recall"],
                "n_edges_best": len(best_dag),
                "elapsed_sec": round(time.time() - t0, 1),
            }
            all_results.append(row)
            log.info(str(row))

    # save best DAG for this seed
    pd.DataFrame(sorted(best_dag), columns=["X", "Y"]).to_csv(
        f"{OUT_DIR}/mimic_best_seed{seed_val}.csv", index=False
    )

log.info("=== Final results table ===")
res_df = pd.DataFrame(all_results)
log.info("\n" + res_df.to_string(index=False))
res_df.to_csv("mimic_alarm_results.csv", index=False)

# summary across seeds at final checkpoint
final_it = MAX_ITERS
summary = res_df[res_df["iteration"] == final_it].agg({"SHD": ["mean", "std"], "SID": ["mean", "std"], "spurious": ["mean", "std"], "precision": ["mean", "std"], "recall": ["mean", "std"], "best_BIC": ["mean", "std"]})
log.info(f"=== Summary at iteration {final_it} across {len(SEEDS)} seeds ===")
log.info("\n" + str(summary))

# reference: BIC of the TRUE DAG, for comparison
true_bic = score_dag(set(true_model.edges()))
log.info(f"True ALARM DAG BIC score: {true_bic:.2f}")
log.info(f"True ALARM DAG edge count: {len(true_model.edges())}")