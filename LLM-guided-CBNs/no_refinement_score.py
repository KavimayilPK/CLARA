"""
no_refinement_score.py
----------------------
CLARA without refinement: take edges files from a folder, build a BayesianNetwork
as-is, score it with StructuredBicScore (and TreeBICScore), and write a JSON file
in the same format that refine_step_1.py produces.

JSON output format (matches refine_step_1.py):
  [
    {
      "llm_type": "...",
      "sample_id": 0,
      "structured_bic": <float>,
      "tree_bic": <float>,
      "edges": [["X", "Y"], ...]
    },
    ...
  ]

Usage:
    python no_refinement_score.py
        --edges_folder  <path to folder with initial_edges_*.csv files>
        --data_file     <path to data CSV>
        --black_list    <path to black_list CSV>
        --output        <path for output JSON>
"""

import argparse
import json
import os
import sys

import pandas as pd
from pathlib import Path
from pgmpy.models import BayesianNetwork

# from structure_learning import StructuredBicScore  # skipped in CLARA loop — causes OOM on dense graphs
from tree_bic_score import TreeBICScore

tld = Path(__file__).parent
sys.path.insert(0, str(tld.parent))   # make project root importable
import domain_config as dc

# --------------------------------------------------------------------------- #
# Defaults (mirror refine_step_1.py layout)
# --------------------------------------------------------------------------- #
DEFAULT_EDGES_FOLDER = os.path.join(tld, "initial_edge_lists")
DEFAULT_OUTPUT       = os.path.join(tld, "no_refinement_results.json")


def load_edges(edges_file: str, data: pd.DataFrame, black_list: list[tuple]) -> list[tuple]:
    """Load edges from CSV, drop blacklisted ones, and skip unknown nodes."""
    df = pd.read_csv(edges_file)

    # Remove blacklisted edges
    black_set = set(black_list)
    df = df[~df.apply(lambda r: (r.X, r.Y) in black_set, axis=1)]

    edges = []
    for _, row in df.iterrows():
        if row.X not in data.columns or row.Y not in data.columns:
            print(f"  Skipping edge ({row.X}, {row.Y}): unknown node.")
        else:
            edges.append((row.X, row.Y))
    return edges


def score_graph(edges: list[tuple], data: pd.DataFrame, state_names: dict) -> dict:
    """Build a BayesianNetwork from edges and return both BIC scores."""
    model = BayesianNetwork()
    model.add_nodes_from(data.columns)
    model.add_edges_from(edges)

    # StructuredBicScore skipped here — it fits exponentially many DecisionTrees
    # for dense nodes (many parents) and causes OOM in the CLARA feedback loop.
    # structured_bic = StructuredBicScore(data, state_names=state_names).score(model)
    structured_bic = float("nan")
    tree_bic       = TreeBICScore(data, state_names=state_names).score(model)

    return {
        "structured_bic": float(structured_bic),
        "tree_bic":       float(tree_bic),
        "edges":          list(edges),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Score LLM edge lists without refinement.")
    p.add_argument("--domain",       default="alarm", choices=list(dc.DOMAINS.keys()),
                   help="Domain — sets default data_file and black_list paths")
    p.add_argument("--edges_folder", default=DEFAULT_EDGES_FOLDER,
                   help="Folder containing initial_edges_*.csv files")
    p.add_argument("--data_file",    default=None,
                   help="Path to the dataset CSV (defaults to domain's dataset)")
    p.add_argument("--black_list",   default=None,
                   help="Path to black_list CSV (defaults to domain's blacklist)")
    p.add_argument("--output",       default=DEFAULT_OUTPUT,
                   help="Path for the output JSON file")
    args = p.parse_args()
    cfg = dc.DOMAINS[args.domain]
    if args.data_file is None:
        args.data_file = str(cfg["dataset"])
    if args.black_list is None:
        args.black_list = str(cfg["blacklist"])
    return args


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Load shared resources
    # ------------------------------------------------------------------ #
    data = pd.read_csv(args.data_file).astype(int)
    print(f"Loaded data: {data.shape[0]} rows, {data.shape[1]} columns.")

    state_names = dc.build_state_names(args.domain, data.columns)

    black_list_df = pd.read_csv(args.black_list)
    black_list = [(row.X, row.Y) for _, row in black_list_df.iterrows()
                  if row.X in data.columns and row.Y in data.columns]
    print(f"Loaded {len(black_list)} blacklist edges.")

    # ------------------------------------------------------------------ #
    # Process each edges file
    # ------------------------------------------------------------------ #
    results = []

    edges_files = sorted(
        f for f in os.listdir(args.edges_folder) if f.startswith("initial_edges")
    )
    if not edges_files:
        print(f"No 'initial_edges*' files found in {args.edges_folder}.")
        return

    for fname in edges_files:
        # Derive llm_type the same way refine_step_1.py does
        llm_type = " ".join(fname.split("_")[2:]).replace(".csv", "")
        print(f"\nProcessing: {fname}  (llm_type={llm_type!r})")

        edges = load_edges(os.path.join(args.edges_folder, fname), data, black_list)
        print(f"  {len(edges)} edges after filtering.")

        scores = score_graph(edges, data, state_names)

        results.append({
            "llm_type":      llm_type,
            "sample_id":     0,
            "structured_bic": scores["structured_bic"],
            "tree_bic":       scores["tree_bic"],
            "edges":          scores["edges"],
        })

        print(f"  structured_bic = {scores['structured_bic']:.4f}")
        print(f"  tree_bic       = {scores['tree_bic']:.4f}")

    # ------------------------------------------------------------------ #
    # Write JSON (append if file already exists, matching refine_step_1.py)
    # ------------------------------------------------------------------ #
    if os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
    else:
        existing = []

    existing.extend(results)

    with open(args.output, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\nWrote {len(results)} result(s) to {args.output}")


if __name__ == "__main__":
    main()
