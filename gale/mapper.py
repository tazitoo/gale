import itertools
import multiprocessing

import gudhi as gd
import kmapper as km
import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def create_mapper(
    X: np.ndarray,
    f: np.ndarray,
    resolution: int,
    gain: float,
    dist_thresh: float,
    clusterer=AgglomerativeClustering(n_clusters=None, linkage="single"),
) -> dict:
    """Runs Mapper on given some data, a filter function, and resolution + gain parameters.

    Args:
        X (np.ndarray): Array of data. For GALE, this is the feature attribution output (n x k), where there are n samples with k feature attributions each.
        f (np.ndarray): Filter (lens) function. For GALE, the predicted probabilities are the lens function.
        resolution (int): Resolution (how wide each window is)
        gain (float): Gain (how much overlap between windows)
        dist_thresh (float): If using AgglomerativeClustering, this sets the distance threshold as (X.max() - X.min())*thresh. Ignored if clusterer is not AgglomerativeClustering
        clusterer (sklearn.base.ClusterMixin, optional): Clustering method from sklearn. Defaults to AgglomerativeClustering(n_clusters=None, linkage="single").

    Returns:
        dict: Dictionary containing the Mapper output
    """

    mapper = km.KeplerMapper(verbose=0)
    cover = km.Cover(resolution, gain)
    clusterer.distance_threshold = (X.max() - X.min()) * dist_thresh
    #clusterer.eps = dist_thresh
    graph = mapper.map(lens=f, X=X, clusterer=clusterer, cover=cover)
    graph["node_attr"] = {}

    for cluster in graph["nodes"]:
        graph["node_attr"][cluster] = np.mean(f[graph["nodes"][cluster]])

    return graph


def create_pd(mapper: dict) -> list:
    """Creates a persistence diagram from Mapper output.

    Args:
        mapper (dict): Mapper output from `create_mapper`

    Returns:
        list: List of the topographical features
    """
    st = gd.SimplexTree()
    node_idx = {}

    for i, n in enumerate(mapper["nodes"].keys()):
        node_idx[n] = i
        st.insert([i])

    for origin in mapper["links"]:
        edges = mapper["links"][origin]

        for e in edges:
            if e != origin:
                st.insert([node_idx[origin], node_idx[e]])
    attrs = {node_idx[k]: mapper["node_attr"][k] for k in mapper["nodes"].keys()}

    for k, v in attrs.items():
        st.assign_filtration([k], v)
    st.make_filtration_non_decreasing()
    st.extend_filtration()
    dgms = st.extended_persistence(min_persistence=1e-5)
    pdgms = []

    for dgm in dgms:
        pdgms += [d[1] for d in dgm]

    return pdgms


def bottleneck_distance(mapper_a: dict, mapper_b: dict) -> float:
    """Calculates the bottleneck distance between two Mapper outputs (denoted A and B)

    Args:
        mapper_a (dict): Mapper A, from `create_mapper`
        mapper_b (dict): Mapper B, from `create_mapper`

    Returns:
        float: the bottleneck distance
    """
    pd_a = create_pd(mapper_a)
    pd_b = create_pd(mapper_b)

    return gd.bottleneck_distance(pd_a, pd_b)

#def one_boot_mapper(X, f, idxs, r, g, d, clusterer):
#    # Randomly select points with replacement
#    Xboot = X[idxs, :]
#    fboot = f[idxs]
#    # Fit mapper
#    M_boot = create_mapper(Xboot, fboot, r, g, d, clusterer)
#
#    return M_boot

# Sub function to run the bootstrap sequence
#def bootstrap_sub(params):
def bootstrap_sub(X, f, idx, r, g, d, clusterer):
    #baseline mapper
    M = create_mapper( X=X, f=f, resolution=r, gain=g, dist_thresh=d, clusterer=clusterer)

    # boostrapped mapper
    X_boot = X[idx, :]
    f_boot = f[idx]
    M_boot = create_mapper(X_boot, f_boot, r, g, d, clusterer)
    G_boot = mapper_to_networkx(M_boot)
    cc = nx.number_connected_components(G_boot)
    bdist = bottleneck_distance(M_boot, M)
    return r, g, d, bdist, cc


def grid_search(
    X: np.ndarray,
    f: np.ndarray,
    resolutions: list,
    gains: list,
    distances: list,
    clusterer=AgglomerativeClustering(n_clusters=None, linkage="single"),
    ci=0.95,
    n_boots=30,
    n_jobs=1,
) -> dict:
    """Bootstraps the data to figure out the best Mapper parameters through a greedy search.

    Args:
        X (np.ndarray): this is the feature attribution output (n, k) where there are n samples with k feature attributions
        f (np.ndarray): Filter (lens) function, typically the predicted probabilities 
        resolutions (list): resolutions to test.
        gains (list): gains to test.
        distances (list): If using AgglomerativeClustering, this sets the distance threshold as (X.max() - X.min())*thresh.
        clusterer (sklearn.base.ClusterMixin, optional): Clustering method from sklearn. Defaults to AgglomerativeClustering(n_clusters=None, linkage="single").
        ci (float, optional): Confidence interval to create. Defaults to 0.95.
        n (int, optional): Number of bootstraps to run. Defaults to 30.
        n_jobs (int, optional): Number of processes for multiprocessing. Defaults to CPU count. -1 for all cores.

    Returns:
        dict: Dictionary containing the Mapper parameters found in a greedy search
    """
    # Create parameter list
    paramlist = list(
        itertools.product(
            [X], [f], resolutions, gains, distances, [clusterer], [ci], [n_boots],[n_jobs]
        )
    )

    # Create MP pool

    if n_jobs < 1:
        pool = multiprocessing.Pool()
    else:
        pool = multiprocessing.Pool(processes=n_jobs)

    results = pool.starmap(bootstrap_sub, paramlist)
    return results


def dictify_results(results):
    # dictify the list of tuples we get as results
    # loop over tuples of r, g, d, and stab and cc
    res_dict = {}
    res = []
    for res in results:
        r, g, d, stab, cc = res
        if (r, g, d) not in res_dict.keys():
            res_dict[(r, g, d)] = {}
            res_dict[(r, g, d)]['stability'] = []
            res_dict[(r, g, d)]['cc'] = []
            res_dict[(r, g, d)]['stability'].append(stab)
            res_dict[(r, g, d)]['cc'].append(cc)
        else:
            res_dict[(r, g, d)]['stability'].append(stab)
            res_dict[(r, g, d)]['cc'].append(cc)

    return res_dict


def post_process(res_dict, ci):
    # loop over keys (r, g, d) candidate tuples
    # find worst case 95th %ile stability and connected components
    # also track max stability and max cc for loss calculation
    max_stability = 0
    max_component = 0
    for a_cand in res_dict.keys():
        # get sorted order for debugging 
#        res_dict[a_cand]['stability'].sort()
#        res_dict[a_cand]['cc'].sort()
        distribution = res_dict[a_cand]['stability']
        cc = res_dict[a_cand]['cc']
        distribution = np.sort(distribution)
        cc = np.sort(cc)
        dist_thresh = distribution[int(ci * len(distribution))]
        cc_thresh = cc[int(ci * len(cc))]

        if max_stability < max(distribution):
            max_stability = max(distribution)
        if max_component < max(cc):
            max_component = max(cc)

        res_dict[a_cand]['dist_thresh'] = dist_thresh
        res_dict[a_cand]['cc_thresh'] = cc_thresh
    return res_dict, max_stability, max_component

def calculate_loss(res_dict, max_stability, max_component):
    # calculate loss over all candidates
    #   rescale so each are similar magnitude
    #   scalarize the two objectives with equal weighting
    #   Calculate distance to (0,0) and take the smallest
    eps = 1e-8
    min_stability = 0  # perfectly stable
    min_component = 1  # everthing is connected
    for a_cand in res_dict.keys():
#        s = np.array(res_dict[a_cand]['stability'])
#        c = np.array(res_dict[a_cand]['cc'])
        s = np.array(res_dict[a_cand]['dist_thresh'])
        c = np.array(res_dict[a_cand]['cc_thresh'])
        stab = (s - min_stability) / (max_stability - min_stability + eps)
        comp = (c - min_component) / (max_component - min_component + eps)
        loss = np.sqrt(stab**2 + comp**2)
        res_dict[a_cand]['loss'] = loss
    return res_dict


def find_best_params(results, ci):
    '''
    Find "best" hyperparameters that minimize our loss.
    then, calculate the distance to (0,0). Select the params with minimum distance.

    Inputs: results - list of results (r, g, d, stab, cc) (e.g. from mp.pool)
    Output: i. tuple of hyperparameters for best candidate
            ii. our dictionary / database of results for validation
    '''

    summary_dict = {}
    res_dict = dictify_results(results)
    res_dict, max_stability, max_component = post_process(res_dict, ci)
    res_dict = calculate_loss(res_dict, max_stability, max_component)

    min_loss = 999
    for a_cand in res_dict.keys():
        loss = res_dict[a_cand]['loss']
        if min_loss > loss.min():
            min_loss = loss.min()
            best_cand = a_cand

    summary_dict['max_stability'] = max_stability
    summary_dict['max_component'] = max_component
    summary_dict['min_loss'] = min_loss
    summary_dict['best_cand'] = best_cand
    return summary_dict, res_dict


def mapper_to_networkx(mapper: dict) -> nx.classes.graph.Graph:
    """Takes the Mapper output (which is a `dict`) and transforms it to a networkx graph.

    Args:
        mapper (dict): Mapper output from `create_mapper`

    Returns:
        nx.classes.graph.Graph: Networkx graph produced by the Mapper output.
    """
    G = nx.Graph()
    node_idx = {}

    for i, n in enumerate(mapper["nodes"].keys()):
        node_idx[n] = i
        G.add_node(i)

    for origin in mapper["links"]:
        edges = mapper["links"][origin]

        for e in edges:
            if e != origin:
                G.add_edge(node_idx[origin], node_idx[e])
    attrs = {
        node_idx[k]: {"avg_pred": mapper["node_attr"][k]}

        for k in mapper["nodes"].keys()
    }
    nx.set_node_attributes(G, attrs)

    return G
