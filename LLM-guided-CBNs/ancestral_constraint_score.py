import math
import networkx as nx
from pgmpy.estimators import StructureScore


class AncestralConstraintScore(StructureScore):
    """
    Wraps any pgmpy StructureScore and returns -inf for any local_score
    call that would result in an ancestral constraint violation.

    An ancestral constraint (X, Y) asserts X must be an ancestor of Y,
    i.e. a directed path X -> ... -> Y must exist. A violation occurs when
    Y can reach X (reversed direction) in the graph.

    How it works
    ------------
    pgmpy's HillClimbSearch keeps a single DAG object and mutates it in-place
    (hill_climb.py line ~298: `current_model = start_dag`). By holding a
    reference to that same object via `live_dag`, this scorer can inspect the
    live graph state on every local_score call — without modifying any
    existing code.

    For each candidate operation the scorer reconstructs a hypothetical graph:
      - the current live graph with node's parents replaced by the proposed ones
      - flip detection: if a proposed parent was previously a child of node, the
        old edge is also removed so the check reflects the true post-flip state

    Usage
    -----
        from pgmpy.models import BayesianNetwork
        from pgmpy.estimators import BicScore
        from ancestral_constraint_score import AncestralConstraintScore

        ancestral_constraints = [
            ("Age",   "Disease"),    # Age must be an ancestor of Disease
            ("Smoke", "Cancer"),
        ]

        M0 = BayesianNetwork()
        M0.add_nodes_from(data.columns)
        M0.add_edges_from(edges)

        base = BicScore(data, state_names=state_names)
        scorer = AncestralConstraintScore(data, base, ancestral_constraints, live_dag=M0)

        # Pass M0 directly — NOT M0.copy() — so scorer and refine_bn share the
        # same object that HillClimbSearch will mutate during search.
        M1 = refine_bn(
            M0, data, state_names,
            required_edges=required_edges,
            scoring_method=scorer,
            tabu_length=0,
            black_list=black_list,
            subtractive_refinement_only=True,
        )

    Notes
    -----
    Caching: HillClimbSearch uses ScoreCache by default, which caches results
    keyed by (node, frozenset(parents)). Because this scorer's output depends
    on the live graph state, caching gives incorrect results when edges are
    added or flipped. For subtractive-only refinement this is harmless (each
    parent set is only evaluated once). For full search (add + flip), pass
    use_cache=False when constructing HillClimbSearch.
    """

    def __init__(self, data, base_score, ancestral_constraints, live_dag):
        """
        Parameters
        ----------
        data : pd.DataFrame
        base_score : StructureScore instance
            Any pgmpy scorer, e.g. BicScore(data), TreeBICScore(data).
        ancestral_constraints : list of (str, str)
            Each (ancestor, descendant) pair asserts ancestor ≺ descendant.
        live_dag : BayesianNetwork or DAG
            The exact DAG object passed as start_dag to HillClimbSearch.
            Must NOT be copied after this call.
        """
        super().__init__(data)
        self.base_score = base_score
        self.constraints = ancestral_constraints
        self.live_dag = live_dag

    # ------------------------------------------------------------------ #
    # pgmpy interface                                                      #
    # ------------------------------------------------------------------ #

    def local_score(self, node, parents):
        """
        Returns -inf if setting `parents` as the parents of `node` would
        violate any ancestral constraint in the resulting graph.
        Otherwise delegates to the base scorer.
        """
        parents = list(parents)

        # Build a graph: take the live graph and swap out
        # node's current parents for the proposed ones.
        G = nx.DiGraph(self.live_dag.edges())

        current_parents = set(G.predecessors(node))
        proposed_parents = set(parents)

        for p in current_parents:
            G.remove_edge(p, node)

        for p in proposed_parents:
            G.add_edge(p, node)

        # Flip detection: if a newly proposed parent was previously a child
        # of node, this is a flip operation — remove the now-reversed edge
        # so the hypothetical graph accurately reflects the post-flip state.
        for p in proposed_parents - current_parents:
            if self.live_dag.has_edge(node, p):
                G.remove_edge(node, p)

        # Check every constraint on the hypothetical graph.
        for (anc, desc) in self.constraints:
            if anc == desc:
                continue
            if anc not in G.nodes or desc not in G.nodes:
                continue
            # Violation: required forward path anc →* desc no longer exists,
            # OR a reverse path desc →* anc was introduced (e.g. by a flip).
            if not nx.has_path(G, anc, desc) or nx.has_path(G, desc, anc):
                return -math.inf

        return self.base_score.local_score(node, parents)

    def structure_prior_ratio(self, operation):
        return self.base_score.structure_prior_ratio(operation)
