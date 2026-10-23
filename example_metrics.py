import networkx as nx
import pickle

def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

# Examples of metrics to supply the auditor
def edge_count(path):
    G = load_pkl(path)
    return G.number_of_edges()


def diameter(path):
    G = load_pkl(path)
    if not nx.is_connected(G):
        return float('inf')
    return nx.diameter(G, usebounds=True)


def assortativity(path):
    G = load_pkl(path)
    return nx.degree_assortativity_coefficient(G)


def triangle_count(path):
    G = load_pkl(path)
    return sum(nx.triangles(G).values())


def num_connected_components(path):
    G = load_pkl(path)
    if isinstance(G, nx.DiGraph):
        return len([*nx.weakly_connected_components(G)])
    else:
        return len([*nx.connected_components(G)])


# distributional / parameterised functions can be used as partials
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


def top_kth_betweenness(path, k=0):
    G = load_pkl(path)
    bc = sorted(nx.betweenness_centrality(G).values(), reverse=True)
    return bc[k]


def top_kth_closeness(path, k=0):
    G = load_pkl(path)
    cc = sorted(nx.closeness_centrality(G).values(), reverse=True)
    return cc[k]


def top_kth_clustering(path, k=0):
    G = load_pkl(path)
    cc = sorted(nx.clustering(G).values(), reverse=True)
    return cc[k]
