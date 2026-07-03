import os
import sys
import pandas as pd
from pathlib import Path
from structure_learning import refine_bn, StructuredBicScore
from pgmpy.readwrite import BIFWriter
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator, BicScore
import json
import datetime

from tree_bic_score import TreeBICScore
from ancestral_constraint_score import AncestralConstraintScore

tld = Path(__file__).parent
results_path = os.path.join(tld, "results")
initial_edges_path = os.path.join(tld, "initial_edge_lists")

# file to store all refined results
refined_result_file = os.path.join(tld, "g1_refined_results.json")

# file to store bic scores for each llm type and bootstrap sample
bic_scores_file = os.path.join(tld, f"bic_scores.csv")
# store all refined results - edges, bic scores, llm type
all_refined_results = []

if __name__ == "__main__":
    import argparse
    sys.path.insert(0, str(tld.parent))   # make project root importable
    import domain_config as dc

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="alarm", choices=list(dc.DOMAINS.keys()),
                        help="Domain to run refinement for")
    parser.add_argument("--no-required-edges", action="store_true",
                        help="Skip required edges — run plain subtractive refinement only")
    args = parser.parse_args()
    domain_cfg = dc.DOMAINS[args.domain]

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(os.path.join(results_path, "llm"), exist_ok=True)
    os.makedirs(os.path.join(results_path, "llm", "subtractive"), exist_ok=True)

    # data structure to store the best bic scores for each llm type
    best_llm_bic = []

    for llm_edges_file in os.listdir(initial_edges_path):
        if llm_edges_file.startswith("initial_edges"):
            llm_type = " ".join(llm_edges_file.split("_")[2:])
            llm_type = llm_type.replace(".csv", "")
            print("Considering the LLM edges from", llm_type)
            initial_edges_df = pd.read_csv(os.path.join(initial_edges_path, llm_edges_file))

            # Remove any edges from initial_edges_df that are present in black_list
            black_list_df = pd.read_csv(domain_cfg["blacklist"])
            print("Removing any black list edges from the initial edges")

            initial_edges_df = initial_edges_df.merge(black_list_df, on=["X", "Y"], how="left", indicator=True)
            initial_edges_df = initial_edges_df[initial_edges_df["_merge"] == "left_only"].drop(columns=["_merge"])

            # data structure to store the refined BIC scores for each bootstrap sample for this llm type
            refined_bic_scores = []

            # Load the domain dataset (single file per domain)
            for dataset_path in [domain_cfg["dataset"]]:
                data = pd.read_csv(dataset_path).astype(int)
                sample_id = 0
                print("\nProcessing bootstrap data for sample", sample_id)

                edges = []
                for _, row in initial_edges_df.iterrows():
                    if (row.X not in data.columns) or (row.Y not in data.columns):
                        print(f"Edge {(row.X, row.Y)} contains an unknown node. Skipping.")
                    else:
                        edges.append((row.X, row.Y))
                print(f"Loaded {len(edges)} edges.")

                black_list = []
                for i, row in black_list_df.iterrows():
                    if (row.X not in data.columns) or (row.Y not in data.columns):
                        print(f"Edge {(row.X, row.Y)} contains an unknown node. Skipping.")
                    else:
                        black_list.append((row.X, row.Y))
                print(f"Loaded {len(black_list)} edges as black list.")

                # load required edges
                if args.no_required_edges:
                    required_edges = []
                    print("Required edges skipped (--no-required-edges).")
                else:
                    required_df = pd.read_csv(os.path.join(tld, "required_edges.csv"))
                    required_edges = []
                    for _, row in required_df.iterrows():
                        if (row.X in data.columns) and (row.Y in data.columns):
                            required_edges.append((row.X, row.Y))
                        else:
                            print(f"Required edge {(row.X, row.Y)} has unknown node. Skipping.")
                    print(f"Loaded {len(required_edges)} required edges.")

                M0 = BayesianNetwork()
                M0.add_nodes_from(data.columns)
                M0.add_edges_from(edges)

                state_names = dc.build_state_names(args.domain, data.columns)

                ancestral_df = pd.read_csv(os.path.join(tld, "ancestral_constraints.csv"))
                ancestral_pairs = [
                    (row.X, row.Y) for _, row in ancestral_df.iterrows()
                    if row.X in data.columns and row.Y in data.columns
                ]

                base_scorer = TreeBICScore(data, state_names=state_names)
                if ancestral_pairs:
                    print(f"Using AncestralConstraintScore with {len(ancestral_pairs)} constraints.")
                    scorer = AncestralConstraintScore(data, base_scorer, ancestral_pairs, live_dag=M0)
                    M2 = refine_bn(M0, data, state_names=state_names, required_edges=required_edges,
                                   scoring_method=scorer, tabu_length=0, black_list=black_list,
                                   subtractive_refinement_only=True)
                else:
                    scorer = base_scorer
                    M2 = refine_bn(M0.copy(), data, state_names=state_names, required_edges=required_edges,
                                   scoring_method=scorer, tabu_length=0, black_list=black_list,
                                   subtractive_refinement_only=True)

                M2.fit(data, state_names=state_names, estimator=BayesianEstimator,
                       prior_type="dirichlet", pseudo_counts=1)

                refined_model_bic = StructuredBicScore(data, state_names=state_names).score(M2)
                refined_tree_bic  = TreeBICScore(data, state_names=state_names).score(M2)
                refined_bic_scores.append((sample_id, refined_tree_bic, refined_model_bic))

                refined_edges_list = list(M2.edges())

                all_refined_results.append({
                    "llm_type": llm_type,
                    "sample_id": sample_id,
                    "structured_bic": float(refined_model_bic),
                    "tree_bic":        float(refined_tree_bic),
                    "edges": refined_edges_list
                })

                BIFWriter(M2).write_bif(os.path.join(results_path,
                                                     "llm", "subtractive", f"{llm_type}_{sample_id}-2.bif"))


            # save the best refined BIC score for this llm type (ranked by tree_bic)
            if refined_bic_scores:
                best_sample, best_tree_bic, best_struct_bic = max(refined_bic_scores, key=lambda x: x[1])
                best_llm_bic.append((llm_type, best_sample, best_tree_bic, best_struct_bic))

    # append to existing json results
    if os.path.exists(refined_result_file):
        with open(refined_result_file, "r") as f:
            existing_results = json.load(f)
    else:
        existing_results = []

    existing_results.extend(all_refined_results)

    with open(refined_result_file, "w") as f:
        json.dump(existing_results, f, indent=2)

    # display all best refined BIC scores for each llm type (ordered by tree_bic)
    print("Summary of best refined BIC scores for each LLM type:")
    for llm_type, sample_id, tree_bic, struct_bic in best_llm_bic:
        print(f"LLM Type: {llm_type}, Sample ID: {sample_id}, Tree BIC: {tree_bic}, Structured BIC: {struct_bic}")

    # save all best refined BIC scores to bic_scores_file
    bic_scores_df = pd.DataFrame(best_llm_bic, columns=["LLM_Type", "Sample_ID", "Best_Tree_BIC", "Best_Structured_BIC"])
    bic_scores_df.to_csv(bic_scores_file, mode="a", index=False)

    # print the best of all LLM graphs (ranked by tree_bic)
    overall_best = None
    if best_llm_bic:
        overall_best = max(best_llm_bic, key=lambda x: x[2])
        print(f"\nOverall Best LLM Type: {overall_best[0]}, Sample ID: {overall_best[1]}, Tree BIC: {overall_best[2]}, Structured BIC: {overall_best[3]}")
