from auditor import DPGRMAuditor
import example_metrics
from functools import partial
import networkx as nx
import pickle


def main():
    # Example: Karate-Club graph pair construction
    G0 = nx.karate_club_graph()
    G1 = nx.karate_club_graph()
    v = max(G1, key=G1.degree)
    G1.remove_edges_from(list(G1.edges(v)))
    print(G0, '\n', G1)

    with open(f"./case_study/Data/Karate-Club/nx_adj0.pkl", 'wb') as f:
        pickle.dump(G0, f)
    with open(f"./case_study/Data/Karate-Club/nx_adj0.txt", 'w') as f:
        for u, v in G0.edges():
            f.write(f"{u} {v}\n")

    with open(f"./case_study/Data/Karate-Club/nx_adj1.pkl", 'wb') as f:
        pickle.dump(G1, f)
    with open(f"./case_study/Data/Karate-Club/nx_adj1.txt", 'w') as f:
        for u, v in G1.edges():
            f.write(f"{u} {v}\n")
    
    # Then, run the mechanism several times to generate output graphs
    # ...
    
    # Run the audit
    n_trials = 400
    epsilon_claimed = 0.1
    delta_claimed = 1e-5
    confidence = 0.99

    # Witness statistics
    metrics = {
        "Edge Count": example_metrics.edge_count,
        "Max. Degree": partial(example_metrics.top_kth_degree, k=0), 
        "Max. PageRank": partial(example_metrics.top_kth_pagerank, k=0),
    }

    stem = f"./case_study/PrivDPR/result/Karate-Club/nx_adj_eps{epsilon_claimed}"
    pairs = [(f"{stem}_i{i}_g0.pkl", f"{stem}_i{i}_g1.pkl") for i in range(n_trials)]

    for metric_name, metric_fn in metrics.items():
        # Audit instantiation
        a = DPGRMAuditor(pairs, epsilon_claimed, delta_claimed, confidence, metric_fn, selection_evaluation_split=0.5, seed=0)
        results = a.run()
        a.plot(metric_name=metric_name, title=f"Karate Club ({metric_name})")
        del a


if __name__ == "__main__":
    main()