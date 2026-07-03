# This is the python file for converting the model with edges that we have to actual bn models
import os
import json
import pandas as pd
from pathlib import Path

tld = Path(__file__).parent.parent
os.makedirs(os.path.join(tld, "results"), exist_ok=True)
os.makedirs(os.path.join(tld, "results", "llm"), exist_ok=True)
os.makedirs(os.path.join(tld, "results", "llm", "no_refinement"), exist_ok=True)

if __name__ == "__main__":
    for file in os.listdir(os.getcwd()):
        if not file.endswith(".csv"):
            continue
        llm_name = file.split("_")[-1].split(".")[0]
        if llm_name == "union" or llm_name == "intersection":
            llm_name = "combined " + llm_name

        print("Saving the edge list of", llm_name, "as a graph")
        df = pd.read_csv(os.path.join(os.getcwd(), file))
        edges = list(df.values.tolist())
        nodes = sorted(set(n for edge in edges for n in edge))
        graph = {"nodes": nodes, "edges": edges}

        with open(os.path.join(tld, "results", "llm", "no_refinement", llm_name + ".graph"), "w") as f:
            json.dump(graph, f, indent=4)


    print("All the graphs have been constructed")



