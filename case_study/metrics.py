import networkx as nx
from utils import load_pkl

def edge_count(path):
    G = load_pkl(path)
    return G.number_of_edges()

def top_kth_degree(path, k=0):
    G = load_pkl(path)
    deg = [d for v, d in G.degree()]
    deg.sort(reverse=True)
    return deg[k]

def top_kth_pagerank(path, k=0):
    G = load_pkl(path)
    if G.number_of_edges() == 0:
        return 1.0 / max(G.number_of_nodes(), 1)
    pr = sorted(nx.pagerank(G, max_iter=5000, tol=1e-12).values(), reverse=True)
    return pr[k]
