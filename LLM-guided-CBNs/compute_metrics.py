# This is the file for computing the metrics for the different models
import copy
import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import BIFReader
from cdt.metrics import SHD, SID
from tqdm import tqdm

# Note, the current code doesn't support more than 10 bootstrap samples due to the way the files are parsed
#num_bootstrap_samples = 10
num_bootstrap_samples = 1
tld = Path(__file__).parent
graph_dir = os.path.join(tld, 'results')

def load_json_graph(graph_path):
    # We have the json file which we need to load
    with open(graph_path, 'r') as f:
        graph = json.load(f)

    # The json file contains the "nodes" and "edges" keys and we need to convert it to a pgmpy graph
    nodes = graph['nodes']
    edges = graph['edges']
    pgmpy_graph = BayesianNetwork()
    pgmpy_graph.add_nodes_from(nodes)
    pgmpy_graph.add_edges_from(edges)
    return pgmpy_graph

def load_bif_graph(graph_path):
    # We have the bif file which we need to load
    bif_model = BIFReader(graph_path)
    return bif_model.get_model()

def compute_metrics(test, expert):
    """
    Compute the metrics for the test graph in relation with the expert graph
    :param test:
    :param expert:
    :return: The SHD, SID and spurious edges respectively for the graph
    """
    # Some learned graphs (e.g. from BIF) omit isolated nodes; add them so SHD can build a full adjacency matrix
    test = copy.deepcopy(test)
    for node in expert.nodes():
        if node not in test.nodes():
            test.add_node(node)
    shd = SHD(copy.deepcopy(expert), test, double_for_anticausal=False)
    sid = SID(copy.deepcopy(expert), test)
    spurious = len(set(test.edges()).difference(set(expert.edges())))
    return shd, sid, spurious

def initialize_metrics(m_types: list, mets: list):
    met = {}
    for m in m_types:
        met[m] = {k: np.zeros(num_bootstrap_samples) for k in mets}
    return met

def compute_metrics_for_mode(mode: str):
    expert_graph = load_json_graph(os.path.join(graph_dir, "expert.graph"))
    if mode == "data":
        model_types = ["pc", "fci1", "fci2"]
        model_path = os.path.join(graph_dir, "data_only")
    elif mode == "llm_subtractive" or mode == "llm_full" or mode == "llm_only":

        model_types = ["gpt5", "gpt51","gpt52","gpt53","gpt54","gpt55","combined union","consensus"]

        if mode == "llm_subtractive":
            model_path = os.path.join(graph_dir, "llm", "subtractive")
        elif mode == "llm_full":
            model_path  = os.path.join(graph_dir, "llm", "full")
        else:
            model_path = os.path.join(graph_dir, "llm", "no_refinement")
    else:
        raise ValueError("Invalid mode")

    metrics = initialize_metrics(m_types=model_types, mets=["shd", "sid", "spurious"])

    for graph_model in tqdm(os.listdir(model_path)):
        if graph_model.endswith('1.bif') and mode != "data":
            continue
        if graph_model.endswith('.graph'):
            test_graph = load_json_graph(os.path.join(model_path, graph_model))
        else:
            test_graph = load_bif_graph(os.path.join(model_path, graph_model))

        # Once we have the test graph, we would compute its metrics in relation with the expert graph
        shd, sid, spurious = compute_metrics(test_graph, expert_graph)

        if mode == "llm_only":
            model_name = graph_model.split(".")[0]
            # We don't have any bootstrap samples for llm only
            metrics[model_name]['shd'] = np.array([shd])
            metrics[model_name]['sid'] = np.array([sid])
            metrics[model_name]['spurious'] = np.array([spurious])
            continue

        if graph_model.endswith('.bif'):
            # gss_0--2.bif  →  prefix="gss_0", variant="2"  →  model_name="gss_2", sample_id=0
            base = graph_model[:-4]
            prefix, variant = base.rsplit('--', 1)
            model_name = prefix.split('_')[0] + '_' + variant
            raw_id = prefix.split('_')[1]
        else:
            model_name = graph_model.split('_')[0]
            raw_id = graph_model.split('_')[1].split('.')[0]
        if not raw_id.isdigit():
            metrics[model_name]['shd'] = np.array([shd])
            metrics[model_name]['sid'] = np.array([sid])
            metrics[model_name]['spurious'] = np.array([spurious])
            continue
        sample_id = int(raw_id)

        # We would update the metrics dict with the computed metrics
        metrics[model_name]['shd'][sample_id] = shd
        metrics[model_name]['sid'][sample_id] = sid
        metrics[model_name]['spurious'][sample_id] = spurious

    # Once we have computed the metrics for all the graphs, we would compute the mean and std of the metrics
    return metrics

def compute_overall_statistics(metrics, save_dir):
    # This is the method for computing the overall metrics for the data
    # like mean, median, std dev as well as p-statistic for the data and what not
    metrics_df = None
    for overall_method_type, val_o in metrics.items():
        for method_or_llm_metrics, val_ml in val_o.items():
            if metrics_df is None:
                cols = ["type", "method"]
                cols_2 = [f"{k}_mean" for k in val_ml.keys()]
                cols_2 += [f"{k}_std" for k in val_ml.keys()]
                cols_2.sort()
                cols = cols + cols_2
                metrics_df = pd.DataFrame({k: [] for k in cols})
            row = {"type": overall_method_type, "method": method_or_llm_metrics}
            for met, val in val_ml.items():
                mean = np.mean(val)
                std = np.std(val)
                row.update({f"{met}_mean": mean, f"{met}_std": std})

            # Once we have the row, we would add it to the dataframe
            row_df = pd.DataFrame(row, index=[0])
            metrics_df = pd.concat([metrics_df, row_df], ignore_index=True)

    metrics_df.to_csv(os.path.join(save_dir, "computed_metrics.csv"), index=False)

metrics_save_path = os.path.join(graph_dir, "metrics.pkl")
compute_from_scratch = True
if __name__ == '__main__':
    if compute_from_scratch:
        # We would call the subroutine for the three modes
        final_metrics = {}
        print("Computing the metrics for data only refinement")
        final_metrics["data_only"] = compute_metrics_for_mode("data")
        print("Computing the metrics for llm no refinement")
        final_metrics["llm_only"] = compute_metrics_for_mode("llm_only")
        #print("Computing the metrics for llm subtractive refinement")
        #final_metrics["llm_subtractive"] = compute_metrics_for_mode("llm_subtractive")
        #print("Computing the metrics for llm full refinement")
        #final_metrics["llm_full"] = compute_metrics_for_mode("llm_full")

        with open(metrics_save_path, 'wb') as f:
            pickle.dump(final_metrics, f)

    else:
        with open(os.path.join(metrics_save_path), "rb") as f:
            final_metrics = pickle.load(f)


    compute_overall_statistics(final_metrics, save_dir=graph_dir)











