from typing import List, Set
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import networkx as nx
import numpy as np
from networkx.algorithms import community
from torch_geometric.utils import to_networkx, from_networkx
from src.datatypes.sparse import from_data

from src.datatypes import reg_dataset_wrapper

from copy import deepcopy


class GraphClusteringException(Exception):
    pass


def find_optimal_resolution(g, max_comm_size=1000, min_res=0.1, max_res=2.0, how_many=30):
    """Optimize the resolution parameter for the Louvain algorithm, given a graph g.
    Optimization involves maximizing modularity, while keeping the size of the largest
    community below max_comm_size.

    Parameters
    ----------
    g : nx.Graph
        networkx graph to 
    max_comm_size : int, optional
        maximum size of biggest community that can be found. A resolution is discarded
        if Louvain returns a bigger community, by default 1000
    min_res : float, optional
        minimum resolution, by default 0.1
    max_res : float, optional
        maximum resolution, by default 2.0
    how_many : int, optional
        number of optimization runs, and number of values considered in [min_res, max_res], by default 30

    Returns
    -------
    Tuple[float, float, np.ndarray, np.ndarray]
        optimal resolution, optimal modularity, resolutions considered, modularities found for resolutions considered
    """
    modularities = []
    resolutions = np.linspace(min_res, max_res, how_many)
    for r in resolutions:
        partition = community.louvain_communities(g, resolution=r, seed=0)
        max_len = max([len(part) for part in partition])
        if max_len > max_comm_size:
            modularity = -1
        else:
            try:
                modularity = community.quality.modularity(g, partition)
            except ZeroDivisionError: # case with partitions with only one node
                modularity = 0
        modularities.append(modularity)
    # highlight maximum
    max_modularity = max(modularities)

    if max_modularity == -1:
        raise GraphClusteringException(
            f'No resolution found in [{min_res}, {max_res}] that respects '
            'max_comm_size={max_comm_size}. Try increasing max_comm_size or max_res.'
        )

    max_res = resolutions[modularities.index(max_modularity)]
    return max_res, max_modularity, resolutions, modularities



def find_closer_res_max_comm(g, res, max_comm_size=1000):
    sizes = find_partition_sizes(g, res)
    if max(sizes) <= max_comm_size:
        return res, sizes
    
    # alternate between increasing and decreasing resolution
    i = 1
    sign = 1.

    while True:
        curr_res = res + sign * i/10.
        sizes = find_partition_sizes(g, curr_res)
        if max(sizes) <= max_comm_size:
            break

        sign *= -1
        if sign == 1:
            i += 1


    return curr_res, sizes


def run_with_multiproc_or_not(worker, iterable, len, n_jobs=1):
    if n_jobs == 1:
        return [worker(i) for i in tqdm(iterable, total=len)]
    else:
        chunksize = 40
        with Pool(n_jobs) as p:
            return list(tqdm(p.imap(worker, iterable, chunksize=chunksize), total=len))


class WorkerFindPartitionSizes:

    def __init__(self, max_comm_size=1000, max_res=2.0, fix_too_large_error=False):
        self.max_comm_size = max_comm_size
        self.max_res = max_res
        self.fix_too_large_error = fix_too_large_error

    def __call__(self, i_g):
        i, g = i_g
        try:
            res, _, _, _ = find_optimal_resolution(g, max_comm_size=self.max_comm_size, max_res=self.max_res)
            partition = list(community.louvain_communities(g, resolution=res, seed=0))
            curr_sizes = [len(part) for part in partition]
        except GraphClusteringException as e:
            print('Too large community found in dataset', i)
            if self.fix_too_large_error:
                res, curr_sizes = find_closer_res_max_comm(g, res, max_comm_size=self.max_comm_size)
                partition = list(community.louvain_communities(g, resolution=res, seed=0))
            else:
                raise e
            
        return curr_sizes, partition

from multiprocessing import Pool

def find_partition_sizes(graphs, max_comm_size=1000, max_res=2.0, fix_too_large_error=False, n_jobs=1):
    # parallelize
    sizes = []
    partitions = []

    worker = WorkerFindPartitionSizes(max_comm_size=max_comm_size, max_res=max_res, fix_too_large_error=fix_too_large_error)

    results = run_with_multiproc_or_not(worker, enumerate(graphs), len(graphs), n_jobs=n_jobs)
    sizes, partitions = zip(*results)

    return sizes, partitions


def create_super_graph(g: nx.Graph, partition: List[Set[int]]):
    """Create a super graph of a graph g given a node partition where:
    - each node in the super graph represents a community in the partition
    - there is an edge between two nodes in the super graph if there is an edge between
        nodes in the two communities in the partition.

    Parameters
    ----------
    g : nx.Graph
        graph to create super graph from
    partition : List[Set[int]]
        list of sets of nodes, each set representing a community, and forming a
        partition of the nodes of g.

    Returns
    -------
    nx.Graph
        supergraph of g, where each node represents a community in the partition.
    """
    # create super graph with nodes
    super_g = nx.Graph()
    for i, part in enumerate(partition):
        super_g.add_node(i, num_nodes=len(part))

    # add edges between communities if there is an edge between nodes in the two communities
    for i in range(len(partition)):
        part_i = list(partition[i])
        for j in range(len(partition)):
            if i == j: continue
            part_j = list(partition[j])
            # if there is an edge between the two communities
            if any(g.has_edge(ni, nj) for ni in part_i for nj in part_j):
                super_g.add_edge(i, j)

    return super_g

def part_sets_to_part_labels(graph: nx.Graph, partition:List[Set[int]]):
    """Convert a partition of nodes into a label for each node in the graph.

    Parameters
    ----------
    graph : nx.Graph
        graph to convert partition to labels
    partition : List[Set[int]]
        list of sets of nodes, each set representing a community, and forming a
        partition of the nodes of g.

    Returns
    -------
    torch.tensor
        tensor of labels for each node in the graph, where the label is the index of the
        community the node belongs to.
    """

    labels = torch.zeros(graph.number_of_nodes(), dtype=torch.int64)
    for i, part in enumerate(partition):
        for n in part:
            labels[n] = i

    return labels

@reg_dataset_wrapper.register()
class CommunityWrapperDataset(Dataset):

    def __init__(self, dataset, max_comm_size=1000, max_res=2.0, n_jobs=1, transform=None):
        super().__init__()

        self.dataset = dataset
        self.transform = transform

        # compute partitions and their max sizes for each graph
        graphs = [to_networkx(d, to_undirected=True) for d in dataset]
        sizes, partitions = find_partition_sizes(graphs, max_comm_size=max_comm_size, max_res=max_res, n_jobs=n_jobs)

        # create super graphs from the node partitions
        super_graphs = [create_super_graph(g, part) for g, part in zip(graphs, partitions)]
        self.super_graphs =[from_data(from_networkx(g, group_node_attrs=['num_nodes'])) for g in super_graphs]

        # get labels for each node in the graph depending on the partition
        self.node_partitions = [part_sets_to_part_labels(g, partition) for g, partition in zip(graphs, partitions)]

        # compute sizes histogram
        sizes_np = np.array(sum(sizes, []))
        sizes_hist, unique_sizes = np.histogram(sizes_np, bins=range(1, max(sizes_np)+2))

        # remove non-support sizes
        support_mask = sizes_hist > 0
        sizes_hist = sizes_hist[support_mask]
        unique_sizes = unique_sizes[:-1][support_mask]
        
        # update stats with sizes histogram
        self.stats = deepcopy(dataset.stats)
        self.stats['sizes_hist'] = {s: h for s, h in zip(unique_sizes, sizes_hist)}


    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, idx):
        g = self.dataset[idx]
        g.node_partition = self.node_partitions[idx]    # set a "node" level parameter
        g.special_supergraph = self.super_graphs[idx]   # set a "special" parameter to avoid batching
        if self.transform is not None:
            g = self.transform(g)
        return g