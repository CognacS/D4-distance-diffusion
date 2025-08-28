from abc import ABC, abstractmethod
from typing import Dict, List, Union, Type

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

from src.datatypes.dense import DenseGraph, DenseEdges, get_node_mask_from_batch
from src.datatypes.sparse import SparseGraph

from copy import deepcopy


class FeatureException(Exception):
    pass


class Feature:

    def apply_added_dims(self, dims: Dict) -> Dict:
        return increase_dims(dims, increase=self.get_added_dims())


    @abstractmethod
    def get_added_dims(self) -> Dict:
        return {}
    


def increase_dims(dims: Dict, increase: Dict) -> Dict:
    dims_copy = deepcopy(dims)

    for k, v in increase.items():
        if k not in dims_copy:
            raise FeatureException(f"Dimension {k} not found in dims when trying to increase it.")
        dims_copy[k] += v

    return dims_copy


def get_dims_list(features: List[Feature]) -> Dict:
    dims = {'x': 0, 'e': 0, 'y': 0}
    for f in features:
        dims = f.apply_added_dims(dims)
    return dims

def increase_dims_list(dims: Dict, features: List[Feature]) -> Dict:
    dims_copy = deepcopy(dims)

    for f in features:
        dims_copy = f.apply_added_dims(dims_copy)

    return dims_copy

    
def source_type_to_target_type(src_type: Type, tgt_type: Type, value: Tensor, src_node_mask=None, tgt_batch=None) -> Tensor:
    if issubclass(src_type, DenseGraph):
        # if no batch is given (nodes num doesn't change)
        # and target is still dense, return the value
        if tgt_batch is None and issubclass(tgt_type, DenseGraph):
            return value
        # else, convert to sparse first
        if src_node_mask is None:
            raise ValueError("Node mask is required to convert from DenseGraph")
        value = value[src_node_mask]
    else:
        if not issubclass(src_type, SparseGraph):
            raise ValueError("Source type must be either DenseGraph or SparseGraph")
    # if source is sparse, then value is already sparse

    # if target is sparse, return the value (sparse)
    if issubclass(tgt_type, SparseGraph):
        return value
    # if target is dense, convert to dense first (to change number of nodes in case)
    elif issubclass(tgt_type, DenseGraph):
        if tgt_batch is None:
            raise ValueError("Batch is required to convert to DenseGraph")
        return to_dense_batch(value, batch=tgt_batch)[0]
    else:
        raise ValueError("Target type must be either DenseGraph or SparseGraph")


def convert_if_mismatch_type(g_src: Union[DenseGraph, SparseGraph], g_tgt: Union[DenseGraph, SparseGraph], value: Tensor) -> Tensor:
    if isinstance(g_src, DenseGraph) and isinstance(g_tgt, SparseGraph):
        if isinstance(g_src, DenseEdges):
            node_mask = get_node_mask_from_batch(g_tgt.batch, g_tgt.num_graphs)
        else:
            node_mask = g_src.node_mask
        return source_type_to_target_type(DenseGraph, SparseGraph, value, src_node_mask=node_mask, tgt_batch=g_tgt.batch)
    elif isinstance(g_src, SparseGraph) and isinstance(g_tgt, DenseGraph):
        raise NotImplementedError("Sparse to Dense conversion is not implemented")
    else:
        return value
    
    
def unpack_graph(graph):
    if isinstance(graph, (SparseGraph, DenseGraph)):
        return ((graph, None),)
    elif isinstance(graph, (tuple, list)):
        if len(graph) == 2:
            return (graph,)
        elif len(graph) == 4:
            return graph[:2], (graph[3], graph[2])
        else:
            raise ValueError(f"Unknown graph format: {graph}, needed 2 or 4 elements")
    else:
        raise ValueError(f"Unknown graph type: {type(graph)}")