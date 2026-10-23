import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from auditor import DPGRMAuditor

import utils, metrics
from functools import partial

def main():
    datasets = {
        "Construction-A": ["d"],#, "u"],
        "Construction-B": ["d"],#, "u"],
        "Construction-C": ["d"],#, "u"],
        "Construction-D": ["d"],#, "u"],
        "RT-LOLGOP": [""], # directed only
        "Anybeat": [""], # directed only
    }
    
    du_code = {
        "d": "(Directed)",
        "u": "(Undirected)",
        "": "",
    }
    
    n_trials = 400
    epsilon_claimed = 0.1
    delta_claimed = 1e-5
    confidence = 0.99
    
    selected_metric = metrics.edge_count
    #selected_metric = partial(metrics.top_kth_degree, k=0)
    #selected_metric = partial(metrics.top_kth_pagerank, k=0)
    
    for dataset, variant in datasets.items():
        stem = f"./PrivDPR/result/{dataset}/nx_adj_eps{epsilon_claimed}"
        for var in variant:
            du = var # directed or undirected
            pairs = [(f"{stem}_i{i}_g0{du}.pkl", f"{stem}_i{i}_g1{du}.pkl") for i in range(n_trials)]
            title = dataset.replace("-", " ")
            print(title)
            a = DPGRMAuditor(pairs, epsilon_claimed, delta_claimed, confidence, selected_metric, selection_evaluation_split=0.5, seed=0)
            results = a.run()
            a.plot(metric_name="Edge Count", title=title)
            del a


if __name__ == "__main__":
    main()