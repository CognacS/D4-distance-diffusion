from typing import Tuple, Dict, Optional
from collections import defaultdict

import torch
from torch import Tensor

from src.datatypes.sparse import SparseGraph
from torch_geometric.utils import scatter

from copy import deepcopy

# same function above, but with "elems" instead of "nodes"
def compute_cum_elems(
        batch: Tensor,
        batch_size: int
    ) -> Tuple[Tensor, Tensor]:
    """Compute the number of elements per graph and the cumulative number of elements per graph in a batch.
    """
    one = batch.new_ones(batch.size(0))
    num_elems = scatter(one, batch, dim=0, dim_size=batch_size, reduce='add')
    cum_elems = torch.cat([batch.new_zeros(1), num_elems.cumsum(dim=0)])
    return num_elems, cum_elems


def build_graphs_batch(
        graph: SparseGraph,
        batch: Tensor,
        batch_size: int,
        original_slice_dict: Dict,
        original_inc_dict: Dict,
        ptr: Optional[Tensor] = None
    ) -> SparseGraph:
    """Fill out batch information from a sparse graph (actually a batch of graphs), the batch index and the batch size.
    This entails:
    1 - fill information into BatchData
    2 - adding the batch index, slice_dict and inc_dict to the BatchData object

    Parameters
    ----------
    graph : Data
        batch of graphs composed of many disconnected graphs
    batch : Tensor
        batch index of each node
    batch_size : int
        size of the considered batch

    Returns
    -------
    Data
        batched version of graph, e.g. BatchData
    """

    # 1 - fill information into BatchData
    # copy slice_dict and inc_dict
    slice_dict, inc_dict = deepcopy(original_slice_dict), deepcopy(original_inc_dict)
    # recall that:
    # slice_dict: for each key, it contains the start and end index of the corresponding data in its datastructure
    # inc_dict: for each key, it contains the increment in node index for each graph in the batch, this is only filled for edge_index

    # fill out slice_dict
    if ptr is None:
        _, cum_nodes = compute_cum_elems(batch, batch_size)
    else:
        cum_nodes = ptr

    # fill node information
    for attr in graph.get_all_node_attrs():
        if attr != 'batch':
            slice_dict[attr] = cum_nodes
            inc_dict[attr] = None

    # fill edge information
    edge_index_batch = batch[graph.edge_index[0]]
    _, cum_edges = compute_cum_elems(edge_index_batch, batch_size)
    slice_dict['edge_index'] = cum_edges
    slice_dict['edge_attr'] = cum_edges
    inc_dict['edge_index'] = cum_nodes[:-1]
    inc_dict['edge_attr'] = None

    for attr in graph.get_all_global_attrs():
        if attr not in slice_dict:
            slice_dict[attr] = torch.arange(batch_size+1, dtype=torch.int64, device=graph.x.device)
            inc_dict[attr] = None

    
    # 2 - adding the batch index, slice_dict and inc_dict to the BatchData object
    graph.batch = batch
    graph.ptr = cum_nodes
    graph._num_graphs = batch_size
    graph._slice_dict = slice_dict
    graph._inc_dict = inc_dict


    return graph