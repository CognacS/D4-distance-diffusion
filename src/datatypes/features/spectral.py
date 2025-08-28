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

        if isinstance(graph, (tuple, list)) and graph[-1] is None:
            graph = graph[0]

        if isinstance(graph, (DenseGraph, SparseGraph)):
            # get adjmat and node_mask from either dense or sparse graph
            adjmat, node_mask = get_adjmat_and_node_mask(g=graph)
            node_masks = node_mask
            batches = None
            
        elif isinstance(graph, (tuple, list)):

            # in this case we have to merge two graphs and the intermediate
            # edges between them

            if len(graph) == 3:
                graph_b, edges_ba, graph_a = graph
                edges_ab = edges_ba.transpose()
            elif len(graph) == 4:
                graph_b, edges_ba, edges_ab, graph_a = graph

            assert isinstance(graph_a, SparseGraph), "graph_a must be a SparseGraph"


            if isinstance(graph_b, DenseGraph):
                # get cumulative nodes
                cum_nodes_t = get_ptr_from_node_mask(graph_b.node_mask)
                batch_b = get_batch_from_node_mask(graph_b.node_mask)
                adjmat_b = graph_b.edge_adjmat
                adjmat_ab = edges_ab.edge_adjmat
                adjmat_ba = edges_ba.edge_adjmat
                if adjmat_b.ndim == 4:
                    adjmat_b = adjmat_b[..., 1:]
                    adjmat_ab = adjmat_ab[..., 1:]
                    adjmat_ba = adjmat_ba[..., 1:]
                # get edge_index
                edge_index_bb, _ = dense_to_sparse(
                        adj=adjmat_b,
                        cum_num_nodes_s=cum_nodes_t,
                        cum_num_nodes_t=cum_nodes_t
                    )
                
                cum_nodes_s = graph_a.ptr

                edge_index_ab, _ = dense_to_sparse(
                    adj=adjmat_ab,
                    cum_num_nodes_s=cum_nodes_s,
                    cum_num_nodes_t=cum_nodes_t
                )
                edge_index_ba, _ = dense_to_sparse(
                    adj=adjmat_ba,
                    cum_num_nodes_s=cum_nodes_t,
                    cum_num_nodes_t=cum_nodes_s
                )
            elif isinstance(graph_b, SparseGraph):
                batch_b = graph_b.batch
                edge_index_bb = graph_b.edge_index
                edge_index_ab = edges_ab.edge_index
                edge_index_ba = edges_ba.edge_index
            else:
                raise ValueError(f"graph_b must be either DenseGraph or SparseGraph, got {type(graph_b)}")

            # graph a is assumed to be sparse
            batch_a = graph_a.batch
            edge_index_aa = graph_a.edge_index

            # merge edge indices
            edge_index = merge_edge_indices(
                edge_index_bb=edge_index_bb,
                edge_index_ab=edge_index_ab,
                edge_index_ba=edge_index_ba,
                edge_index_aa=edge_index_aa,
                num_nodes_a=graph_a.num_nodes,
            )
            # merge batches
            batch = torch.cat([batch_a, batch_b], dim=0)
            # reorder edge_index and batch
            perm = torch.argsort(batch)
            batch = batch[perm]
            edge_index = relabel_edge_index(perm, edge_index)

            num_nodes_per_sample = graph_a.num_nodes_per_sample + graph_b.num_nodes_per_sample

            # get adjmat and node_mask of the merged graph
            adjmat, node_mask = edge_index_batch_to_adjmat_mask(
                edge_index=edge_index,
                batch=batch,
                num_nodes_per_sample= num_nodes_per_sample,
                batch_size=graph_a.num_graphs
            )

            node_mask_a = get_node_mask_from_num_nodes(graph_a.num_nodes_per_sample, max_num_nodes=adjmat.shape[1])
            node_mask_b = get_node_mask_from_num_nodes(num_nodes_per_sample, max_num_nodes=adjmat.shape[1])
            node_mask_b = node_mask_b * (~node_mask_a) # filter out nodes from graph_a

            node_masks = [node_mask_b, node_mask_a]
            batches = [batch_b, batch_a]

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