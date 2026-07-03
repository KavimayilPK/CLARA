import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Set, FrozenSet, DefaultDict

# Script to create history files from the refinment reults. 
# Full History, Top-5 Refined graphs, common edges across top-5 refined graphs, find duplicate edgesets across all refined graphs.

def edge_set_key(edges: List[Any]) -> FrozenSet[Tuple[Any, Any]]:
    # Convert each edge to tuple, then to a frozenset => ignores order and duplicates
    return frozenset(tuple(e) for e in edges)


def find_duplicate_graphs_by_edge_set(results: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Returns groups of result entries that have identical SETS of edges
    (duplicates inside an edge list do not matter).
    """
    groups: Dict[FrozenSet[Tuple[Any, Any]], List[Dict[str, Any]]] = {}
    for r in results:
        k = edge_set_key(r.get("edges", []))
        groups.setdefault(k, []).append(r)

    # keep only groups with 2+ graphs
    return [grp for grp in groups.values() if len(grp) >= 2]


def write_edges_sorted_by_bic(json_path: str, out_txt_path: str) -> None:
    """
    Writes ALL graphs sorted by tree_bic descending (higher is better).
    (kept in spirit with your original function; now also includes rank/sample/bic.)
    """
    # Load JSON
    with open(json_path, "r") as f:
        results = json.load(f)

    if not results:
        raise ValueError(f"No entries found in {json_path}")

    # Sort high -> low by tree_bic
    results_sorted = sorted(results, key=lambda r: float(r["tree_bic"]), reverse=True)

    # Write text output
    with open(out_txt_path, "w") as f:
        for rank, r in enumerate(results_sorted, start=1):
            llm_type = r.get("llm_type", "UNKNOWN")
            sample_id = r.get("sample_id", "NA")
            bic = float(r.get("tree_bic", float("-inf")))

            # JSON stores tuples as lists; convert back to tuples for display
            edges = r.get("edges", [])
            edges_tuples = [tuple(e) for e in edges]

            f.write(f"Rank: {rank}\n")
            f.write(f"Graph: {llm_type}\n")
            f.write(f"Sample: {sample_id}\n")
            f.write(f"BIC: {bic}\n")
            f.write(repr(edges_tuples) + "\n\n")

    print(f"Edges written to: {out_txt_path}")

def pick_top_k_unique_by_edge_set(results: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    """
    Top-k by tree_bic descending (higher is better), only UNIQUE graphs where uniqueness
    is defined by identical edge SET (duplicates ignored).
    """
    results_sorted = sorted(results, key=lambda r: float(r["tree_bic"]), reverse=True)

    seen: Set[FrozenSet[Tuple[Any, Any]]] = set()
    unique_top: List[Dict[str, Any]] = []

    for r in results_sorted:
        key = edge_set_key(r.get("edges", []))
        if key in seen:
            continue
        seen.add(key)
        unique_top.append(r)
        if len(unique_top) >= k:
            break

    if len(unique_top) < k:
        raise ValueError(f"Requested top {k} unique graphs, but only found {len(unique_top)} unique edge sets.")

    return unique_top


def compute_common_edges_across_graphs(graphs: List[Dict[str, Any]]) -> Set[Tuple[Any, Any]]:
    """
    Edges that occur in ALL provided graphs (duplicates ignored).
    """
    if not graphs:
        return set()

    edge_sets = [set(edge_set_key(g.get("edges", []))) for g in graphs]
    return set.intersection(*edge_sets) if edge_sets else set()


def edges_to_adjacency_list(edges: Set[Tuple[Any, Any]]) -> Dict[Any, List[Any]]:
    """
    Convert directed edges (u, v) -> adjacency list u: [v1, v2, ...]
    """
    adj: DefaultDict[Any, Set[Any]] = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)

    # stable sorting even if node types differ
    return {u: sorted(vs, key=lambda x: str(x)) for u, vs in adj.items()}


def write_adjacency_list(adj: Dict[Any, List[Any]], out_path: str) -> None:
    with open(out_path, "w") as f:
        for u in sorted(adj.keys(), key=lambda x: str(x)):
            neighbors = ", ".join(map(str, adj[u]))
            f.write(f"{u} : {neighbors}\n")


def write_top_k_unique_summary(
    results: List[Dict[str, Any]],
    k: int,
    out_txt_path: str,
) -> List[Dict[str, Any]]:
    """
    Writes a TOP-K UNIQUE (by edge-set) history file, similar to your 'top 5 history' notion.
    Returns the selected top-k unique entries as a list for downstream use.
    """
    top_unique = pick_top_k_unique_by_edge_set(results, k=k)

    with open(out_txt_path, "w") as f:
        for rank, r in enumerate(top_unique, start=1):
            llm_type = r.get("llm_type", "UNKNOWN")
            sample_id = r.get("sample_id", "NA")
            bic = float(r.get("tree_bic", float("-inf")))

            edges = r.get("edges", [])
            edges_tuples = [tuple(e) for e in edges]

            f.write(f"Rank: {rank}\n")
            f.write(f"Graph: {llm_type}\n")
            f.write(f"Sample: {sample_id}\n")
            f.write(f"BIC: {bic}\n")
            f.write(repr(edges_tuples) + "\n\n")

    print(f"Top {k} unique graphs written to: {out_txt_path}")
    return top_unique


# ----------------------------
# MAIN
# ----------------------------

if __name__ == "__main__":
    # Assumes g1_refined_results.json is in the same parent folder as this script
    tld = Path(__file__).parent
    #json_path = os.path.join(tld, "LLM-guided-CBNs", "g1_refined_results.json")
    json_path = os.path.join(tld, "LLM-guided-CBNs", "no_refinement_results.json")

    # folder to store results from history creation
    history_dir = os.path.join(tld, "history")
    os.makedirs(history_dir, exist_ok=True)

    with open(json_path, "r") as f:
        results = json.load(f)

    # Keep duplicate reporting
    dup_groups = find_duplicate_graphs_by_edge_set(results)
    if dup_groups:
        print("Duplicate groups (identical edge sets):")
        for i, grp in enumerate(dup_groups, start=1):
            llms = [r.get("llm_type", "UNKNOWN") for r in grp]
            print(f"  Group {i}: {', '.join(llms)}")
    else:
        print("No duplicate edge sets found.")

    # Keep original "write all history" output
    out_all_path = os.path.join(tld, history_dir, "feedback_history_all.txt")
    write_edges_sorted_by_bic(json_path, out_all_path)

    # Add "top 5 history" output (unique graphs)
    out_top5_path = os.path.join(tld, history_dir, "feedback_top5.txt")
    top5_unique = write_top_k_unique_summary(results, k=5, out_txt_path=out_top5_path)

    # edges that occur in ALL of top 5 UNIQUE graphs -> adjacency list
    common_edges = compute_common_edges_across_graphs(top5_unique)
    adj = edges_to_adjacency_list(common_edges)

    out_adj_path = os.path.join(tld, history_dir, "common_edges_top5.txt")
    write_adjacency_list(adj, out_adj_path)

    print(f"Common edges across top 5 unique: {len(common_edges)}")
    print(f"Adjacency list written to: {out_adj_path}")

    # write common edges to a separate .csv file required_edges.csv inside LLM-guided-CBNs folder
    out_csv_path = os.path.join(tld, "LLM-guided-CBNs", "required_edges.csv")
    with open(out_csv_path, "w") as f:
        f.write("X,Y\n")
        for u, v in sorted(common_edges, key=lambda x: (str(x[0]), str(x[1]))):
            f.write(f"{u},{v}\n")
        print(f"Common edges written to: {out_csv_path}")
