"""
eval_clara_outputs.py — convert edge-list CSVs to graphs and compute SHD/SID/spurious/precision/recall.

Usage:
    python eval_clara_outputs.py                        # process all CSVs in clara_eval/, auto-detect domain
    python eval_clara_outputs.py --domain alarm         # override domain for all files
    python eval_clara_outputs.py --input_dir my_dir     # use a different input directory
    python eval_clara_outputs.py --output results.csv   # custom output file name

Drop CSV files (columns: X, Y) into clara_eval/ and run the script.
Domain is inferred from the filename pattern: final_{variant}_{domain}_{config}_{date}_{time}.csv
"""

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
from pgmpy.models import BayesianNetwork
from cdt.metrics import SHD, SID

ROOT = Path(__file__).parent

# Metrics included in aggregation (mean + std)
AGG_METRICS = ["shd", "sid", "spurious", "precision", "recall"]


def load_expert_graph(domain: str) -> BayesianNetwork:
    expert_path = ROOT / "domains" / domain / "expert.graph"
    with open(expert_path) as f:
        data = json.load(f)
    g = BayesianNetwork()
    g.add_nodes_from(data["nodes"])
    g.add_edges_from(data["edges"])
    return g


def csv_to_graph(csv_path: Path) -> BayesianNetwork:
    df = pd.read_csv(csv_path)
    edges = df.values.tolist()
    nodes = sorted(set(n for edge in edges for n in edge))
    g = BayesianNetwork()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def parse_filename(filename: str) -> dict | None:
    """Extract variant, domain, config from filenames like final_{variant}_{domain}_{config}_{date}_{time}.csv"""
    stem = Path(filename).stem
    if not stem.startswith("final_"):
        return None
    parts = stem[len("final_"):].split("_")
    # Strip YYYYMMDD_HHMMSS timestamp from anywhere in the parts list
    filtered = []
    i = 0
    while i < len(parts):
        if (i + 1 < len(parts)
                and len(parts[i]) == 8 and parts[i].isdigit()
                and len(parts[i + 1]) == 6 and parts[i + 1].isdigit()):
            i += 2  # skip both timestamp parts
        else:
            filtered.append(parts[i])
            i += 1
    core = filtered
    if len(core) < 3:
        return None
    return {"variant": core[0], "domain": core[1], "config": "_".join(core[2:])}


def detect_domain(filename: str, override: str | None) -> str | None:
    if override:
        return override
    parsed = parse_filename(filename)
    if parsed and parsed["domain"] in ("alarm", "apo"):
        return parsed["domain"]
    name = filename.lower()
    if "_alarm_" in name:
        return "alarm"
    if "_apo_" in name:
        return "apo"
    return None


def compute_metrics(test: BayesianNetwork, expert: BayesianNetwork) -> dict:
    shd = SHD(copy.deepcopy(expert), test, double_for_anticausal=False)
    sid = SID(copy.deepcopy(expert), test)
    expert_edges = set(map(tuple, expert.edges()))
    test_edges = set(map(tuple, test.edges()))
    spurious = len(test_edges - expert_edges)
    missing = len(expert_edges - test_edges)
    retrieved = len(test_edges & expert_edges)
    recall = retrieved / len(expert_edges) if expert_edges else 0.0
    precision = retrieved / (retrieved + spurious) if (retrieved + spurious) > 0 else 0.0
    return {
        "shd": shd,
        "sid": sid,
        "spurious": spurious,
        "missing": missing,
        "retrieved_edges": retrieved,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLARA edge-list CSVs")
    parser.add_argument("--input_dir", default="clara_eval",
                        help="Directory containing input CSV files (default: clara_eval/)")
    parser.add_argument("--domain", choices=["alarm", "apo"],
                        help="Override domain for all files")
    parser.add_argument("--output", default="clara_eval_metrics.csv",
                        help="Output CSV path (default: clara_eval_metrics.csv)")
    args = parser.parse_args()

    input_dir = ROOT / args.input_dir
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    expert_cache: dict[str, BayesianNetwork] = {}
    rows = []

    for csv_path in csv_files:
        domain = detect_domain(csv_path.name, args.domain)
        if domain is None:
            print(f"  SKIP {csv_path.name} — cannot detect domain (use --domain to override)")
            continue

        if domain not in expert_cache:
            expert_cache[domain] = load_expert_graph(domain)
        expert = expert_cache[domain]

        parsed = parse_filename(csv_path.name)
        variant = parsed["variant"] if parsed else "unknown"
        config = parsed["config"] if parsed else "unknown"

        print(f"  {csv_path.name}  [{domain}]", end="  ")
        try:
            test = csv_to_graph(csv_path)
            for node in expert.nodes():
                if node not in test:
                    test.add_node(node)
            m = compute_metrics(test, expert)
            print(f"SHD={m['shd']}  SID={m['sid']}  spurious={m['spurious']}  "
                  f"missing={m['missing']}  retrieved={m['retrieved_edges']}  "
                  f"precision={m['precision']}  recall={m['recall']}")
            rows.append({"file": csv_path.name, "domain": domain, "variant": variant, "config": config, **m})
        except Exception as e:
            print(f"ERROR: {e}")
            rows.append({
                "file": csv_path.name, "domain": domain, "variant": variant, "config": config,
                **{k: None for k in ["shd", "sid", "spurious", "missing", "retrieved_edges", "precision", "recall"]},
            })

    if not rows:
        print("No files were processed.")
        return

    results_df = pd.DataFrame(rows)

    # Summary: mean + std of AGG_METRICS grouped by (variant, domain, config)
    group_cols = ["variant", "domain", "config"]
    agg_rows = []
    for group_key, group_df in results_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, group_key))
        row["n_runs"] = len(group_df)
        for met in AGG_METRICS:
            vals = pd.to_numeric(group_df[met], errors="coerce").dropna()
            row[f"{met}_mean"] = round(float(vals.mean()), 4) if len(vals) else None
            row[f"{met}_std"] = round(float(vals.std(ddof=0)), 4) if len(vals) > 1 else 0.0
        agg_rows.append(row)
    summary_df = pd.DataFrame(agg_rows)

    out_path = input_dir / args.output
    stem, suffix = out_path.stem, out_path.suffix
    per_file_path = out_path.with_name(stem + "_per_file" + suffix)
    summary_path = out_path.with_name(stem + "_summary" + suffix)

    results_df.to_csv(per_file_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\nPer-file results  → {per_file_path}")
    print(f"Summary (mean/std) → {summary_path}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
