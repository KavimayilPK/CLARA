import os
import pandas as pd
import time
import json
from pathlib import Path
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from statsmodels.multivariate.factor_rotation import target_rotation

tld = Path(__file__).parent.parent
data_path = os.path.join(tld, "datasets")
results_path = os.path.join(tld, "results")

if __name__ == "__main__":
    data_file = "apo_data.csv"
    sample_id = "0"
    domain = "apo"
    data = pd.read_csv(os.path.join(data_path, data_file)).astype(int)
    print("Learning from FCI for", data_file)
    time.sleep(1)

    background_knowledge = None
    if domain == "apo":
        bl_df = pd.read_csv(os.path.join(tld, "black_list_apo.csv"))
        node_map = {col: GraphNode(col) for col in data.columns}
        background_knowledge = BackgroundKnowledge()
        for _, row in bl_df.iterrows():
            if row["X"] in node_map and row["Y"] in node_map:
                background_knowledge.add_forbidden_by_node(node_map[row["X"]], node_map[row["Y"]])
        print(f"  Loaded {len(bl_df)} forbidden edges for APO")

    g, _ = fci(data.to_numpy(), independence_test_method="chisq", background_knowledge=background_knowledge)

    # Rename the nodes and edges base on the data
    pdy = GraphUtils.to_pydot(g, labels=data.columns)

    # Mapping from internal node to label name
    node_label_map = {}
    for node in pdy.get_nodes():
        name = node.get_name().strip('"')
        label = node.get_attributes().get("label", name).strip('"')
        node_label_map[name] = label

    # Extracting edges using actual labels
    pdy_edges_direct = []
    pdy_edges_weak_direct = []
    all_edges = []

    for edge in pdy.get_edges():
        ignore_edge = -1
        # We would only consider directly causal edges
        edge_ends = (edge.obj_dict["attributes"]["arrowhead"], edge.obj_dict["attributes"]["arrowtail"])
        if not (edge_ends[0] == "normal" and edge_ends[1] == "none"):
            if not (edge_ends[0] == "normal" and edge_ends[1] == "odot"):
                ignore_edge = 1
            else:
                ignore_edge = 0
        source = str(edge.get_source())
        target = str(edge.get_destination())
        source_label = node_label_map[source]
        target_label = node_label_map[target]
        if ignore_edge == -1:
            pdy_edges_direct.append((source_label, target_label))
        elif ignore_edge == 0:
            pdy_edges_weak_direct.append((source_label, target_label))
        all_edges.append((source_label, target_label, (edge_ends[1], edge_ends[0])))

    # Save the graph as a json file
    graph_json_fci1 = {
        "nodes": list(data.columns),
        "edges": pdy_edges_direct,
        "_all_edges": all_edges
    }

    graph_json_fci2 = {
        "nodes": list(data.columns),
        "edges": pdy_edges_weak_direct + pdy_edges_direct,
        "_all_edges": all_edges
    }

    with open(os.path.join(results_path, "data_only", f"fci1_{sample_id}.graph"), "w") as f:
        json.dump(graph_json_fci1, f, indent=3)

    with open(os.path.join(results_path, "data_only", f"fci2_{sample_id}.graph"), "w") as f:
        json.dump(graph_json_fci2, f, indent=3)
    print(f"  Saved fci1_{sample_id}.graph and fci2_{sample_id}.graph")
