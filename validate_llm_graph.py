import networkx as nx
import pandas as pd
import os
import json

# validate initial graphs (G0) generated from LLM. 

# Define file paths
base_dir = os.path.dirname(os.path.abspath(__file__))
blacklist_file = os.path.join(base_dir, "black_list_post.csv")
expert_graph = os.path.join(base_dir, "expert.graph")

# Active expert-graph path — overridden at runtime by set_expert_graph()
_expert_graph_path = expert_graph


def set_expert_graph(path) -> None:
    """Point check_variables() at a domain-specific expert graph.

    Call this once in run_clara() before any graph generation starts.
    Pass None (or a non-existent path) to skip node validation entirely.
    """
    global _expert_graph_path
    _expert_graph_path = str(path) if path else None

# construct a graph from llm provided edges
def construct_llm_graph(edges):
    """
    Returns:
    Graph object
    """
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G

# check if the graph is a valid DAG
def is_valid_dag(llm_graph):      
    """ 
    Check if the given LLM graph is a valid DAG (Acyclicity).
    Parameters:
    llm_graph: A directed graph (networkx DiGraph).

    Returns:
    None if the graph is valid (without cycles), List of cycles otherwise.
    """
    if nx.is_directed_acyclic_graph(llm_graph):
        return None
    else:
        cycles = list(nx.simple_cycles(llm_graph))
        return cycles
    
def check_variables(nodes):
    """
    Check if the nodes in LLM graph are present in the expert graph.

    Parameters:
    nodes: List of variable names.

    Returns:
    List of nodes that are irrelevant.
    None if all nodes are present.
    """
    if _expert_graph_path is None or not os.path.exists(_expert_graph_path):
        print("Expert graph not found or not configured — node validation skipped.")
        return None
    with open(_expert_graph_path, 'r', encoding='utf-8') as f:
        expert_data = json.load(f)

    expert_nodes = expert_data.get("nodes", [])

    dataset_nodes = [n.strip() for n in expert_nodes]

    llm_nodes = [str(n).strip() for n in nodes if pd.notna(n)]
    
    invalid_nodes = [llm_node for llm_node in llm_nodes if llm_node not in dataset_nodes]

    if invalid_nodes:
        print("Invalid nodes: ",invalid_nodes)
        return invalid_nodes
    else:
        return None

def check_black_list(edges):
    """
    Check if any edges in LLM graph are in the blacklist.
    
    Parameters:
    edges: List of directed edges as tuples (source, target).
    
    Returns:
    List of edges that are blacklisted.
    None if no blacklisted edges are found.
    """
    #Read blacklist edges from a csv file 
    blacklist_df = pd.read_csv(blacklist_file)
    blacklist = list(zip(blacklist_df['X'], blacklist_df['Y']))

    blacklisted_edges = [edge for edge in edges if edge in blacklist]

    if blacklisted_edges:
        print("Blacklisted Edges Present:",blacklisted_edges)
        return blacklisted_edges
    else:
        print("There are no blacklisted edges")
        return None

# Usage example
if __name__ == "__main__":

    # get list of llm edges files from initial_llm_edges directory
    llm_edges_dir = os.path.join(base_dir, "initial_llm_edges")
    llm_edges_files = [f for f in os.listdir(llm_edges_dir) if f.startswith("initial_edges_") and f.endswith(".csv")]

    for llm_edges_file in llm_edges_files:
        llm_edges_path = os.path.join(llm_edges_dir, llm_edges_file)
        print(f"Validating LLM edges from file: {llm_edges_path}")

        # Read LLM edges from CSV file
        llm_edges_df = pd.read_csv(
            llm_edges_path,
            usecols=['X', 'Y'],
            dtype={'X': str, 'Y': str},
            keep_default_na=False,)
        
        llm_edges_df["X"] = llm_edges_df["X"].str.strip()
        llm_edges_df["Y"] = llm_edges_df["Y"].str.strip()

        # keep only rows that have BOTH endpoints - in this way we check for isolated nodes in graph
        llm_edges_df = llm_edges_df[(llm_edges_df["X"] != "") & (llm_edges_df["Y"] != "")]

        llm_edges = list(zip(llm_edges_df["X"], llm_edges_df["Y"]))

        # Draw LLM graph
        llm_graph = construct_llm_graph(llm_edges)

        # check if the LLM graph satisfies all conditions
        invalid_nodes = check_variables(llm_graph.nodes())
        cycles = is_valid_dag(llm_graph)
        blacklisted_edges = check_black_list(llm_edges)

        if not invalid_nodes and not cycles and not blacklisted_edges:
            #print in green color
            print(f"\033[92m{llm_edges_file} is valid and satisfies all conditions.\033[0m")
        else:
            # print in differnt color
            print(f"\033[91m{llm_edges_file} is invalid and does not satisfy all conditions.\033[0m")

    
    