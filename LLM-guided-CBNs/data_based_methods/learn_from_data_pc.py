# This is the method for learning the data using PGMPy PC algorithm
import os
import pandas as pd
import json
from pathlib import Path
from pgmpy.estimators import PC, ExpertKnowledge
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork

tld = Path(__file__).parent.parent
data_path = os.path.join(tld, "datasets")
results_path = os.path.join(tld, "results")

if __name__ == '__main__':
    data_file = "apo_data.csv"
    sample_id = "0"
    domain = "apo"  # set to "apo" to use APO dataset with blacklist
    try:
        data = pd.read_csv(os.path.join(data_path, data_file)).astype(int)
        print(f"\nProcessing {data_file} using PC algorithm")

        expert_knowledge = None
        if domain == "apo":
            bl_df = pd.read_csv(os.path.join(tld, "black_list_apo.csv"))
            forbidden = list(zip(bl_df["X"], bl_df["Y"]))
            expert_knowledge = ExpertKnowledge(forbidden_edges=forbidden)
            print(f"  Loaded {len(forbidden)} forbidden edges for APO")

        c = PC(data)
        dag = c.estimate(return_type="dag", expert_knowledge=expert_knowledge)
        bn = BayesianNetwork()
        bn.add_nodes_from(data.columns)
        if dag is not None:
            for u, v in dag.edges():
                try:
                    bn.add_edge(u, v)
                except ValueError:
                    print(f"  Skipping cycle-forming edge {u}->{v}")

        output_file = os.path.join(results_path, "data_only", f"pc_{sample_id}.graph")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        graph = {"nodes": list(bn.nodes()), "edges": list(bn.edges())}
        with open(output_file, 'w') as f:
            json.dump(graph, f, indent=3)
        print(f"  Saved to pc_{sample_id}.graph")
    except Exception as e:
        print(f"  ERROR ({type(e).__name__}): {e}")
