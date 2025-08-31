from typing import Tuple, Union
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_dense_batch

from src.datatypes.features.core import Feature, source_type_to_target_type
from src.datatypes.features.posenc import SinusoidalPosEmb
from src.datatypes.dense import (
    DenseGraph,
    DenseEdges,
    get_node_mask_from_batch,
    dense_to_sparse,
    get_batch_from_node_mask,
    get_ptr_from_node_mask,
    get_node_mask_from_num_nodes
)
from src.datatypes.split import merge_edge_indices, relabel_edge_index
from src.datatypes.sparse import SparseGraph

from src.datatypes.features import reg_features

from src.datatypes.spectral import (
    get_eigenfeatures_from_adjmat,
    get_node_cycle_features_from_adjmat
)


def edge_index_batch_to_adjmat_mask(edge_index, batch, num_nodes_per_sample, batch_size):
    adjmat = to_dense_adj(edge_index, batch=batch, batch_size=batch_size)
    node_mask = get_node_mask_from_batch(
        batch=batch,
        batch_size=batch_size,
        num_nodes=num_nodes_per_sample
    )
    return adjmat, node_mask


def get_adjmat_and_node_mask(g: Union[DenseGraph, SparseGraph]) -> Tuple[Tensor, Tensor]:
    if isinstance(g, DenseGraph):
        adjmat = g.edge_adjmat
        node_mask = g.node_mask
    elif isinstance(g, SparseGraph):
        adjmat, node_mask = edge_index_batch_to_adjmat_mask(
            edge_index=g.edge_index,
            batch=g.batch,
            num_nodes_per_sample=g.num_nodes_per_sample,
            batch_size=g.num_graphs
        )
    return adjmat, node_mask


def cat_feature(graph, name, value, node_masks, batches):

    if isinstance(graph, (list, tuple)):
        if name == 'x':
            value_a = source_type_to_target_type(DenseGraph, type(graph[0]), value, node_masks[0], tgt_batch=batches[0])
            value_b = source_type_to_target_type(DenseGraph, type(graph[-1]), value, node_masks[-1], tgt_batch=batches[-1])
        else:
            value_a, value_b = value, value

        setattr(graph[0], name, torch.cat([getattr(graph[0], name), value_a], dim=-1))
        setattr(graph[-1], name, torch.cat([getattr(graph[-1], name), value_b], dim=-1))
    else:
        if name == 'x':
            value = source_type_to_target_type(DenseGraph, type(graph), value, node_masks)
        setattr(graph, name, torch.cat([getattr(graph, name), value], dim=-1))


@reg_features.register('spectral')
class SpectralFeature(Feature):

    def __init__(self, mode='all', cycles=True, encode_ints=True, encoded_dim= 8, encoded_scale=1.0, topk_eigvals=5, topk_eigvecs=2, **kwargs):
        super().__init__()
        self.mode = mode
        self.cycles = cycles
        self.topk_eigvals = topk_eigvals
        self.topk_eigvecs = topk_eigvecs
        self.encode_ints = encode_ints
        self.encoded_dim = encoded_dim if encode_ints else 1
        if self.encode_ints:
            self.emb = SinusoidalPosEmb(self.encoded_dim, scale=encoded_scale)
        self.encoder = lambda x: self.emb(x).flatten(start_dim=-2) if self.encode_ints else x
        

    def get_added_dims(self):
        shapes = {'x': 0, 'y': 0}
        if self.mode == 'all' or self.mode == 'eigenvalues':
            shapes['y'] += self.topk_eigvals + self.encoded_dim
        if self.mode == 'all':
            shapes['x'] += self.topk_eigvecs + 1
        if self.cycles:
            shapes['x'] += self.encoded_dim * 3
            shapes['y'] += self.encoded_dim * 4

        return shapes

    def __call__(self, graph: Data) -> Data:

        # get adjmat and node_mask from either dense or sparse graph
        adjmat, node_mask = get_adjmat_and_node_mask(g=graph)
        node_masks = node_mask
        batches = None

        if self.cycles:
            # calculate cyclefeatures
            cyclefeatures = get_node_cycle_features_from_adjmat(
                adjmat=adjmat, node_mask=node_mask
            )
            # unpack cyclefeatures
            x_cycles, y_cycles = cyclefeatures

            # concatenate features to graph
            if x_cycles is not None:
                cat_feature(graph, 'x', self.encoder(x_cycles), node_masks, batches)
            if y_cycles is not None:
                cat_feature(graph, 'y', self.encoder(y_cycles), node_masks, batches)


        if self.mode is not None and self.mode != 'none':
            # calculate eigenfeatures
            eigenfeatures = get_eigenfeatures_from_adjmat(
                adjmat=adjmat, node_mask=node_mask,
                mode=self.mode, topk_eigvals=self.topk_eigvals,
                topk_eigvecs=self.topk_eigvecs
            )
            # unpack eigenfeatures
            n_connected_comp, batch_eigenvalues, nonlcc_indicator, k_lowest_eigenvector = eigenfeatures

            # concatenate features to graph
            if n_connected_comp is not None:
                cat_feature(graph, 'y', self.encoder(n_connected_comp), node_masks, batches)
            if batch_eigenvalues is not None:
                cat_feature(graph, 'y', batch_eigenvalues, node_masks, batches)
            if nonlcc_indicator is not None:
                cat_feature(graph, 'x', nonlcc_indicator, node_masks, batches)
            if k_lowest_eigenvector is not None:
                cat_feature(graph, 'x', k_lowest_eigenvector, node_masks, batches)