from typing import Union

import torch
from torch import LongTensor, IntTensor
from src.datatypes.sparse import SparseGraph
from src.datatypes.dense import DenseGraph

#########################  SMALL REPEATED OPERATIONS  ##########################
# the following methods are meant to abstract away some small operations that
# are repeated in the code

def append_time_to_graph_globals(
        graph: Union[DenseGraph, SparseGraph],
        time: Union[IntTensor, LongTensor],
        emb = None
    ) -> Union[DenseGraph, SparseGraph]:
    """Append the time to the graph globals vector y
    with the following criteria:
    - if the graph has no y, set y = time
    - if the graph has y, set y = [time, y], that is,
        the time is appended to the beginning of the vector

    Parameters
    ----------
    graph : Union[DenseGraph, SparseGraph]
        any kind of graph with batched y vector of size [batch_size, *] or None
    time : Union[IntTensor, LongTensor]
        time tensor of size [batch_size], this method will unsqueeze to [batch_size, 1]

    Returns
    -------
    same_graph : Union[DenseGraph, SparseGraph]
        same graph as the input, but with the updated y vector
    """

    if emb is None:
        time = time.float().unsqueeze(-1)
    else:
        time = emb(time) # positional embedding

    if graph.y is None:
        graph.y = time
    else:
        if graph.y.ndim == 1:
            graph.y = graph.y.unsqueeze(-1)
        graph.y = torch.cat([time, graph.y], dim = -1)

    return graph


def change_time_in_graph_globals(
        graph: Union[DenseGraph, SparseGraph],
        time: Union[IntTensor, LongTensor],
        emb = None
    ) -> Union[DenseGraph, SparseGraph]:
    """Append the time to the graph globals vector y
    with the following criteria:
    - if the graph has no y, set y = time
    - if the graph has y, set y = [time, y], that is,
        the time is appended to the beginning of the vector

    Parameters
    ----------
    graph : Union[DenseGraph, SparseGraph]
        any kind of graph with batched y vector of size [batch_size, *] or None
    time : Union[IntTensor, LongTensor]
        time tensor of size [batch_size], this method will unsqueeze to [batch_size, 1]

    Returns
    -------
    same_graph : Union[DenseGraph, SparseGraph]
        same graph as the input, but with the updated y vector
    """

    if emb is not None:
        if time.ndim > 1:
            time = time.squeeze(-1) # squeeze to 1D for encoding
        time = emb(time) # positional embedding
    elif time.ndim == 1:
        time = time.unsqueeze(-1)

    time_size = time.shape[-1]
    graph.y[..., :time_size] = time

    return graph