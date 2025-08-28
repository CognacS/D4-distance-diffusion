from typing import List, Dict

from src.data.utils.core import get_dict_histogram

from src.datatypes.sparse import SparseGraph
from src.data.datasets.core import ProcessedDataset

def get_torch_graphs_stats(graphs: List[SparseGraph]|ProcessedDataset) -> Dict:
    """Function for computing general statistics on a list of SparseGraphs.
    Currently returns:
    -  number of classes
    """
    
    l_num_nodes = []
    l_num_edges = []

    for g in graphs:
        l_num_nodes.append(g.num_nodes)
        l_num_edges.append(g.num_edges)

    ret_dict = {
        'num_nodes_min': min(l_num_nodes),
        'num_nodes_max': max(l_num_nodes),
        'num_nodes_hist': get_dict_histogram(l_num_nodes),
        'num_edges_min': min(l_num_edges),
        'num_edges_max': max(l_num_edges),
        'num_edges_hist': get_dict_histogram(l_num_edges),
        'num_graphs': len(graphs)
    }

    return ret_dict