from __future__ import annotations

from typing import Tuple, List, Optional, Union, Callable, Dict, Any

import torch
from torch import Tensor, BoolTensor, IntTensor
import torch.nn.functional as F

from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch, remove_self_loops, dense_to_sparse, scatter, cumsum
from src.datatypes.utils import one_hot

from src.datatypes.sparse import SparseGraph, SparseEdges
from src.datatypes.core import Graph

from copy import copy

################################################################################
#                                  DENSE GRAPH                                 #
################################################################################
class DenseGraphException(Exception):
    pass

class DenseGraph(Graph):
    def __init__(
            self,
            x: Tensor=None,
            edge_adjmat: Tensor=None,
            y: Tensor=None,
            node_mask: BoolTensor=None,
            edge_mask: Optional[Tensor]=None,
            masked=False,
            **kwargs
        ):
        
        super().__init__(
            x = x,
            edge_adjmat = edge_adjmat,
            y = y,
            node_mask = node_mask,
            edge_mask = edge_mask,
            **kwargs
        )
        self.masked = masked
        self.collapsed = self.check_collapsed()
    
    # def to(self, device: str) -> DenseGraph:
    #     """ Changes the device of x, edge_adjmat, y. """
    #     ret_graph = DenseGraph(
    #         x = 			None if self.x is None else self.x.to(device),
    #         edge_adjmat = 	None if self.edge_adjmat is None else self.edge_adjmat.to(device),
    #         y =				None if self.y is None else self.y.to(device),
    #         node_mask =		None if self.node_mask is None else self.node_mask.to(device),
    #         edge_mask =		None if self.edge_mask is None else self.edge_mask.to(device),
    #     )

    #     ret_graph.masked = self.masked
    #     ret_graph.collapsed = self.collapsed

    #     return ret_graph

    def __setattr__(self, key: str, value: Any):
        """if x or edge_adjmat are set, then set masked to False"""
        if self.is_node_attr(key) or self.is_edge_attr(key):
            self.masked = False
        super().__setattr__(key, value)
        if key in ['x', 'edge_adjmat'] and value is not None:
            self.collapsed = self.check_collapsed()


    def check_collapsed(self):
        if hasattr(self, 'x') and self.x is not None:
            return self.x.ndim <= 2
        elif hasattr(self, 'edge_adjmat') and self.edge_adjmat is not None:
            return self.edge_adjmat.ndim <= 3
        else:
            return True



    def type_as(self, x: Tensor):
        """ Changes the device and dtype of x, edge_adjmat, y. """
        if self.x is not None:
            self.x = 			self.x.type_as(x)
        if self.edge_adjmat is not None:
            self.edge_adjmat = 	self.edge_adjmat.type_as(x)
        if self.y is not None:
            self.y = 			self.y.type_as(x)
        return self


    def set_node_mask(self, node_mask: BoolTensor):
        self.node_mask = node_mask
    
    def set_edge_mask(self, edge_mask: Tensor):
        self.edge_mask = edge_mask

    
    def _apply_mask_nodes(self) -> DenseGraph:

        for key in self.get_all_node_attrs():
            
            if key == 'node_mask':
                continue
            
            value = getattr(self, key)
            if value.ndim == 2:
                setattr(self, key, value * self.node_mask)
            elif value.ndim == 3:
                setattr(self, key, value * self.node_mask.unsqueeze(-1))
            else:
                raise DenseGraphException(f'Invalid dimension for node attribute {key}: {value.ndim}')

        return self

    
    def _apply_mask_edges(self) -> DenseGraph:
        
        # get edge mask dense if not set
        self.get_edge_mask_dense()
        
        for key in self.get_all_edge_attrs():
            
            if key == 'edge_mask':
                continue
            
            value = getattr(self, key)
            if value is not None:
                if value.ndim == 3:
                    setattr(self, key, value * self.edge_mask)
                elif value.ndim == 4:
                    setattr(self, key, value * self.edge_mask.unsqueeze(-1))
                else:
                    raise DenseGraphException(f'Invalid dimension for edge attribute {key}: {value.ndim}')

        return self

    def has_node_attrs(self) -> bool:
        return len(self.get_all_node_attrs()) > int(hasattr(self, 'node_mask'))
    
    def has_edge_attrs(self) -> bool:
        return len(self.get_all_edge_attrs()) > int(hasattr(self, 'edge_mask'))

    def apply_mask(self) -> DenseGraph:

        if self.has_node_attrs():
            if self.node_mask is None:
                raise DenseGraphException(
                    'Trying to mask nodes in dense graph, but it has no node mask set. Call set_mask() to set it'
                )
        else:
            if self.has_edge_attrs() and self.edge_mask is None and self.node_mask is None:
                raise DenseGraphException(
                    'Trying to mask edges in dense graph, but it has no edge mask set nor node mask. Call set_mask() to set it'
                )
        
        if not self.masked:
            self._apply_mask_nodes()._apply_mask_edges()
            self.masked = True

        return self


    def get_edge_mask_dense(self) -> Tensor:
        if not hasattr(self, 'edge_mask') or self.edge_mask is None:
            self.edge_mask = get_edge_mask_dense(node_mask=self.node_mask)

        return self.edge_mask


    # def clone(self) -> DenseGraph:
    #     ret_graph = DenseGraph(
    #         x = 			None if self.x is None else self.x.clone(),
    #         edge_adjmat = 	None if self.edge_adjmat is None else self.edge_adjmat.clone(),
    #         y =				None if self.y is None else self.y.clone(),
    #         node_mask =		None if self.node_mask is None else self.node_mask.clone(),
    #         edge_mask =		None if self.edge_mask is None else self.edge_mask.clone()
    #     )

    #     ret_graph.masked = self.masked
    #     ret_graph.collapsed = self.collapsed

    #     return ret_graph
    
    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if Graph.is_global_attr(key):
            if value.ndim == 1 and len(value) == 1:
                return 0
            else:
                return None
        self.keys()

        return super().__cat_dim__(key, value, *args, **kwargs)

    
    # def __getitem__(self, idx) -> Any:
        
    #     if isinstance(idx, str):
    #         return getattr(self, idx)

    #     # graph = DenseGraph(
    #     #     x =             None if self.x is None else self.x[idx].unsqueeze(0),
    #     #     edge_adjmat =   None if self.edge_adjmat is None else self.edge_adjmat[idx].unsqueeze(0),
    #     #     y =             None if self.y is None else self.y[idx].unsqueeze(0),
    #     #     node_mask =     self.node_mask[idx].unsqueeze(0),
    #     #     edge_mask =     self.get_edge_mask_dense()[idx].unsqueeze(0),
    #     #     masked =        self.masked
    #     # )
        
    #     attrs = {}
    #     for key in self.keys():
    #         attrs[key] = getattr(self, key)[idx].unsqueeze(0)
            
    #     graph = DenseGraph(**attrs, masked=self.masked, collapsed=self.collapsed)

    #     return graph
    
    @property
    def num_graphs(self) -> int:
        if self.x is not None:
            return self.x.shape[0]
        elif self.edge_adjmat is not None:
            return self.edge_adjmat.shape[0]
        else:
            return self.y.shape[0]
    
    @property
    def num_nodes(self) -> int:
        return self.node_mask.sum().item()
    
    @property
    def num_nodes_per_sample(self) -> Tensor:
        return self.node_mask.sum(dim=-1)
    
    @property
    def outdegree(self) -> Tensor:
        """Get the outdegree of the graph or batch of graphs."""
        self.apply_mask()
        if self.edge_adjmat is None:
            if self.x is not None:
                return torch.zeros(self.x.shape[0], self.x.shape[1], device=self.device)
            else:
                return 0
        return compute_degree_adjmat(self.edge_adjmat, self.collapsed, inout='out')
    
    @property
    def indegree(self) -> Tensor:
        """Get the indegree of the graph or batch of graphs."""
        self.apply_mask()
        if self.edge_adjmat is None:
            if self.x is not None:
                return torch.zeros(self.x.shape[0], self.x.shape[1], device=self.device)
            else:
                return 0
        return compute_degree_adjmat(self.edge_adjmat, self.collapsed, inout='in')


    def transpose(self):
        new_graph = copy(self)
        for key in self.get_all_edge_attrs():
            value = getattr(new_graph, key)
            setattr(new_graph, key, value.transpose(1, 2))

        new_graph.masked = self.masked

        return new_graph


    
    def _add_if_not_none(self, name, value, l):
        if value is not None:
            l.append(f'{name}={list(value.shape)}, ')
    
    def __repr__(self):

        l = []
        d = [
            *[(key, getattr(self, key)) for key in self.get_all_node_attrs()],
            *[(key, getattr(self, key)) for key in self.get_all_edge_attrs()],
            *[(key, getattr(self, key)) for key in self.get_all_global_attrs()],
            *[(key, getattr(self, key)) for key in self.get_all_special_attrs()]
        ]
        for k, v in d:
            self._add_if_not_none(k, v, l)

        return f'{self.__class__.__name__}({"".join(l)}masked={self.masked}, collapsed={self.collapsed})'
    

def compute_degree_adjmat(edge_adjmat: Tensor, collapsed: bool, inout='in') -> Tensor:

    if inout == 'in':
        dim = 1
    elif inout == 'out':
        dim = 2
    else:
        raise ValueError(f'Invalid value for inout: {inout}')
    
    if collapsed:
        return torch.sum(edge_adjmat > 0, dim=dim)
    else:
        return torch.sum(edge_adjmat[..., 1:].sum(-1), dim=dim)
    



class DenseEdges(DenseGraph):

    def __init__(
            self,
            edge_adjmat: Tensor=None,
            edge_mask: Optional[Tensor] = None,
            masked: bool = False,
            **kwargs
        ):
        super().__init__(
            edge_adjmat = edge_adjmat,
            edge_mask = edge_mask,
            masked = masked,
            **kwargs
        )


class DenseNodes(DenseGraph):

    def __init__(
            self,
            x: Tensor,
            node_mask: Optional[Tensor] = None,
            masked: bool = False,
            indegree: Optional[Tensor] = None,
            outdegree: Optional[Tensor] = None,
            **kwargs
        ):
        super().__init__(
            x = x,
            node_mask = node_mask,
            masked = masked,
            **kwargs
        )
        self._indegree = indegree
        self._outdegree = outdegree


    @property
    def outdegree(self) -> Tensor:
        """Get the outdegree of the graph or batch of graphs."""
        if self._outdegree is None:
            return torch.zeros(self.num_nodes, device=self.x.device)
        else:
            return self._outdegree
    
    @property
    def indegree(self) -> Tensor:
        """Get the indegree of the graph or batch of graphs."""
        if self._indegree is None:
            return torch.zeros(self.num_nodes, device=self.x.device)
        else:
            return self._indegree

    

################################################################################
#                         DENSE GRAPH UTILITY METHODS                          #
################################################################################


###############################  SPARSE TO DENSE  ##############################

def to_dense_adj_bipartite(
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_s: Optional[Tensor] = None,
        batch_t: Optional[Tensor] = None,
        num_nodes_s: Optional[int] = None,
        num_nodes_t: Optional[int] = None,
        max_num_nodes_s: Optional[int] = None,
        max_num_nodes_t: Optional[int] = None,
        batch_size: int = None,
        handle_one_hot: bool = False,
        return_edge_mask: bool = False
    ) -> Tensor:
    """Convert the edge_index and edge_attr for edges from a source graph to a
    target graph to a dense adjacency matrix. This method is an adaptation of
    the to_dense_adj method from the PyTorch Geometric library, taking into account
    the fact that the source and target graphs are different. This way, the
    adjacency matrix is of shape (B, N_s, N_t, F) instead of (B, N, N, F), i.e.
    a square matrix.

    Parameters
    ----------
    edge_index : Tensor
        Tensor (2, E) containing the node indices of the source and target nodes
    edge_attr : Tensor
        Tensor (E, F) containing the edge attributes
    batch_s : Optional[Tensor], optional
        Tensor (N_s,) containing the batch example index for each node in the source
        graph, by default None
    batch_t : Optional[Tensor], optional
        Tensor (N_t,) containing the batch example index for each node in the target
        graph, by default None
    max_num_nodes_s : Optional[int], optional
        maximum number of nodes of the source graph examples in the batch, by default None
    max_num_nodes_t : Optional[int], optional
        maximum number of nodes of the target graph examples in the batch, by default None
    batch_size : int, optional
        number of examples in the batch, by default None

    Returns
    -------
    Tensor
        Tensor (B, N_s, N_t, F) containing the dense adjacency matrix for the batch
    """

    if num_nodes_s is None and num_nodes_t is None and batch_s is None and batch_t is None:
        # if nothing is given
        # setup initial variables
        num_nodes_s = int(edge_index[0].max()) + 1 if edge_index.numel() > 0 else 0
        batch_s = edge_index.new_zeros(num_nodes_s)
        num_nodes_t = int(edge_index[1].max()) + 1 if edge_index.numel() > 0 else 0
        batch_t = edge_index.new_zeros(num_nodes_t)
        batch_size = 1
        
    elif num_nodes_s is None and num_nodes_t is None and batch_s is not None and batch_t is not None:
        # if num_nodes_s and num_nodes_t are not given, but batch_s and batch_t are

        # function for computing num_nodes for batch_s and batch_t, with type hinting
        def compute_num_nodes(batch: Tensor) -> Tuple[Tensor, Tensor]:
            one = batch.new_ones(batch.size(0))
            num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='add')
            return num_nodes
        
        num_nodes_s = compute_num_nodes(batch_s)
        num_nodes_t = compute_num_nodes(batch_t)

    elif batch_s is None and batch_t is None and num_nodes_s is not None and num_nodes_t is not None:
        # if batch_s and batch_t are not given, but num_nodes_s and num_nodes_t are
        batch_s = torch.arange(batch_size).repeat_interleave(num_nodes_s)
        batch_t = torch.arange(batch_size).repeat_interleave(num_nodes_t)


    # compute cumulative number of nodes for each example in the batch
    cum_nodes_s = torch.cat([batch_s.new_zeros(1), num_nodes_s.cumsum(dim=0)])
    cum_nodes_t = torch.cat([batch_t.new_zeros(1), num_nodes_t.cumsum(dim=0)])
    
    # for each edge:
    # idx0: batch index
    idx0 = batch_s[edge_index[0]]
    # idx1: row index (source node)
    idx1 = edge_index[0] - cum_nodes_s[batch_s][edge_index[0]]
    # idx2: column index (target node)
    idx2 = edge_index[1] - cum_nodes_t[batch_t][edge_index[1]]
    

    # get maximum number of nodes for both source and target
    if max_num_nodes_s is None:
        max_num_nodes_s = int(num_nodes_s.max())
    if max_num_nodes_t is None:
        max_num_nodes_t = int(num_nodes_t.max())
    
    # set edge_attr to ones if not given
    if edge_attr is None:
        edge_attr = torch.ones(idx0.numel(), device=edge_index.device)


    # setup size of the adjacency matrix
    size = [batch_size, max_num_nodes_s, max_num_nodes_t]
    size += list(edge_attr.size())[1:]
    flattened_size = batch_size * max_num_nodes_s * max_num_nodes_t

    # idx0 gives the batch index, idx1 the row index, idx2 the column index
    idx = idx0 * max_num_nodes_s * max_num_nodes_t + idx1 * max_num_nodes_t + idx2
    if handle_one_hot:
        # pad one_hot vectors, e.g.: [1, 0, 0] -> [0, 1, 0, 0]
        # this creates a new class for the "no edge" case
        edge_attr = F.pad(edge_attr, (1, 0))
        # allocate the adj matrix
        adj = torch.zeros(flattened_size, edge_attr.size(-1), device=edge_index.device)
        adj[:, 0] = 1
        #adj = F.pad(edge_mask.float().flatten().unsqueeze(-1), (0, edge_attr.shape[-1]-1))

        # fill up the adj matrix with the edge_attr
        adj[idx, :] = edge_attr
        size[-1] += 1
    else:
        # build adjacency matrix by putting edge_attr in the right place
        adj = scatter(edge_attr, idx, dim=0, dim_size=flattened_size, reduce='sum')
    adj = adj.view(size)

    if return_edge_mask:
        node_mask_s = get_node_mask_from_batch(batch_s, batch_size, num_nodes_s, max_num_nodes_s)
        node_mask_t = get_node_mask_from_batch(batch_t, batch_size, num_nodes_t, max_num_nodes_t)
        edge_mask = get_bipartite_edge_mask_dense(node_mask_s, node_mask_t)
        return adj, edge_mask
    else:
        return adj


##############################  DENSE TO SPARSE  ###############################

def dense_to_sparse(
        adj: Tensor,
        cum_num_nodes_s: Optional[Tensor] = None,
        cum_num_nodes_t: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
    """Converts a dense bipartite adjacency matrix into a sparse edge index and
    edge attributes. The number of returned edges is equal to the number of
    non-zero entries in the adjacency matrix.

    Parameters
    ----------
    adj : Tensor
        Tensor (B, N_s, N_t, F), adjacency matrix for the edges between the source and target graphs
    cum_num_nodes_s : Optional[Tensor], optional
        cumulative number of nodes for each example in the source graph batch, by default None
    cum_num_nodes_t : Optional[Tensor], optional
        cumulative number of nodes for each example in the target graph batch, by default None

    Returns
    -------
    edge_index : Tensor
        Tensor (2, E) containing the node indices of the source and target nodes
    edge_attr : Tensor
        Tensor (E, F) containing the edge attributes

    Raises
    ------
    ValueError
        Dense adjacency matrix 'adj' must be 2- or 3-dimensional (got {adj.dim()} dimensions)
    """
    
    if adj.dim() < 2:
        raise ValueError(f"Dense adjacency matrix 'adj' must be 2- or "
                         f"3- or 4-dimensional (got {adj.dim()} dimensions)")

    if adj.dim() == 4:
        adj_collapsed = adj.sum(dim=-1)
    else:
        adj_collapsed = adj

    edge_index = adj_collapsed.nonzero().t()

    if edge_index.size(0) == 2:
        edge_attr = adj[edge_index[0], edge_index[1]]
        return edge_index, edge_attr
    else:
        if cum_num_nodes_s is None:
            node_idx_offset_s = adj.size(1) * edge_index[0]
        else:
            node_idx_offset_s = cum_num_nodes_s[edge_index[0]]

        if cum_num_nodes_t is None:
            node_idx_offset_t = adj.size(2) * edge_index[0]
        else:
            node_idx_offset_t = cum_num_nodes_t[edge_index[0]]

        edge_attr = adj[edge_index[0], edge_index[1], edge_index[2], ...]
        row = edge_index[1] + node_idx_offset_s
        col = edge_index[2] + node_idx_offset_t
        return torch.stack([row, col], dim=0), edge_attr


###############  NO EDGE HANDLING  ###############

def encode_no_edge(
        edge_features: Tensor,
        sparse: bool,
        collapsed: bool
    ) -> Tensor:

    if sparse:
        if collapsed:
            edge_features = edge_features + 1
        else:
            edge_features = torch.cat([edge_features, torch.ones(edge_features.size(0), 1, device=edge_features.device)], dim=-1)
        
    else:
        if collapsed:
            pass
        else:
            raise NotImplementedError

    return edge_features


def remove_no_edge(
        edge_features: Tensor,
        sparse: bool,
        collapsed: bool
    ) -> Tensor:

    if sparse:
        if collapsed:
            edge_features = edge_features - 1
        else:
            edge_features = edge_features[..., 1:]
    
    else:
        if collapsed:
            raise NotImplementedError
        else:
            edge_features = edge_features[..., 1:]

    return edge_features


################  DIRECTIONALITY  ################

def dense_to_undirected(adjmat: Tensor, reduce: str='upper') -> Tensor:

    assert reduce in ['upper', 'lower', 'add', 'mean'], f"Invalid value for reduce: {reduce}"

    if adjmat.dim() == 4:
        adjmat = adjmat.permute(0, 3, 1, 2)

    if reduce == 'upper':
        adjmat = torch.triu(adjmat, diagonal=1)
    elif reduce == 'lower':
        adjmat = torch.tril(adjmat, diagonal=-1)
    
    adjmat = adjmat + adjmat.transpose(-2, -1)

    if reduce == 'mean':
        adjmat = adjmat / 2

    if adjmat.dim() == 4:
        adjmat = adjmat.permute(0, 2, 3, 1)

    return adjmat


##################  CONVERSIONS  #################

from src.datatypes.batch import build_graphs_batch


def to_dense_with_node_mask(
        x: Tensor,
        node_mask: BoolTensor,
        fill_value: float = 0.0
    ):
    batch_size, max_num_nodes = node_mask.shape[:2]
    size = [batch_size, max_num_nodes] + list(x.size())[1:]
    
    out = torch.as_tensor(fill_value, device=x.device)
    out = out.to(x.dtype).repeat(size)
    out[node_mask] = x
    #out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])
    return out


def sparse_graph_to_dense_graph(sparse_graph: SparseGraph, handle_one_hot=False) -> DenseGraph:

    # 1 - build dense node features tensor and node mask (if batch)
    node_mask = None
    batch_size = None
    node_attrs = {}
    if hasattr(sparse_graph, 'batch') and sparse_graph.batch is not None:
        batch = sparse_graph.batch
        batch_size = sparse_graph.num_graphs

        x, node_mask = to_dense_batch(
            x =		sparse_graph.x,
            batch =	batch,
            batch_size=batch_size
        )
        
        node_attrs['x'] = x
        
        for key in sparse_graph.get_all_node_attrs():
            if key != 'x' and key != 'batch':
                value = getattr(sparse_graph, key)
                value = to_dense_with_node_mask(value, node_mask)
                node_attrs[key] = value

        max_num_nodes = x.size(1)
        
    else:
        batch = None
        x: Tensor = sparse_graph.x.unsqueeze(0)
        for key in sparse_graph.get_all_node_attrs():
            if key != 'x' and key != 'batch':
                node_attrs[key] = getattr(sparse_graph, key).unsqueeze(0)
        max_num_nodes = x.size(1)

    # 2- preprocess edges
    edge_index, edge_attr = remove_self_loops(
        edge_index =	sparse_graph.edge_index,
        edge_attr =		sparse_graph.edge_attr
    )

    edge_mask = None
    if handle_one_hot:
        edge_mask = get_edge_mask_dense(node_mask)

    # 3 - build dense adjacency matrix
    edge_attrs = {}
    for key in sparse_graph.get_all_edge_attrs():
        if key == 'edge_index':
            continue
        
        value = getattr(sparse_graph, key)
        
        handle_one_hot_curr = False
        if key == 'edge_attr':
            key = 'edge_adjmat'
            handle_one_hot_curr = handle_one_hot

        value = to_dense_adj_bipartite(
            edge_index =		edge_index,
            edge_attr =			value,
            batch_s =			batch,
            batch_t =			batch,
            max_num_nodes_s =	max_num_nodes,
            max_num_nodes_t =	max_num_nodes,
            batch_size =		batch_size,
            handle_one_hot =    handle_one_hot_curr
        )
        
        edge_attrs[key] = value

    # 4 - build dense target vector
    global_attrs = {}
    for key in sparse_graph.get_all_global_attrs():
        value = getattr(sparse_graph, key)
        if value is not None:
            if value.dim() == 1:
                value = value.unsqueeze(0)
            global_attrs[key] = value

    # 5 - build final dense graph with nodes, edges and targets
    # (optional node mask if it should be a batch)
    dense_graph = DenseGraph(
        **node_attrs,
        **edge_attrs,
        **global_attrs,
        node_mask =		node_mask,
        edge_mask =		edge_mask
    )

    return dense_graph


def sparse_edges_to_dense_edges(sparse_edges: SparseEdges, handle_one_hot=False) -> DenseEdges:

    edge_index, edge_attr = remove_self_loops(
        edge_index =    sparse_edges.edge_index,
        edge_attr =     sparse_edges.edge_attr
    )

    # 3 - build dense adjacency matrix
    edge_adjmat, edge_mask = to_dense_adj_bipartite(
        edge_index =        edge_index,
        edge_attr =         edge_attr,
        num_nodes_s =       sparse_edges.num_nodes_s,
        num_nodes_t =       sparse_edges.num_nodes_t,
        batch_size =        sparse_edges.num_nodes_s.size(0),
        handle_one_hot =    handle_one_hot,
        return_edge_mask =  True
    )

    dense_edges = DenseEdges(
        edge_adjmat =   edge_adjmat,
        edge_mask =     edge_mask
    ).apply_mask()

    return dense_edges


def _subset_adjmatrix(adj: Tensor, node_mask: BoolTensor):
    """Notice: the adjacency matrix and node_mask refer to one single
    sample, not a batch of samples, so adj is 2D, and node_mask is 1D
    """
    # prepare mask
    num_true_nodes = node_mask.sum()
    mask_2d = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)

    # apply mask and reshape as a square matrix
    return adj[mask_2d].reshape(num_true_nodes, num_true_nodes)


def _dense_to_sparse_single_graph(x: Tensor, adj: Tensor, y: Tensor, node_mask: Tensor):

    # the following two operations are done in order to avoid skipping
    # indices during assignment
    # remove fake nodes
    sparse_x = x[node_mask]

    # remove fake edges
    adj = _subset_adjmatrix(adj, node_mask)

    # transform dense adj to edge_index and edge_attr
    edge_index, edge_attr = dense_to_sparse(adj=adj)

    return Data(
        x =				sparse_x,
        edge_index =	edge_index,
        edge_attr =		edge_attr,
        y =				y
    )


def dense_remove_self_loops(
        adjmat: Tensor
    ) -> Tensor:
    """Modifies inplace the adjacency matrix by removing self loops"""

    mask = torch.eye(adjmat.shape[1], dtype=torch.bool, device=adjmat.device)

    # if adjmat is a batch of adjacency matrices
    if adjmat.shape[1] == adjmat.shape[2]:
        mask = mask.unsqueeze(0).expand(adjmat.shape[0], -1, -1)

    # remove self loops
    adjmat[mask] = 0

    return adjmat


def dense_graph_to_sparse_graph(
        dense_graph: DenseGraph,
        num_nodes: IntTensor,
        batchify: bool = False
    ) -> SparseGraph:
    
    device = dense_graph.edge_adjmat.device

    attrs = {}
    # collect node tensors
    for attr in dense_graph.get_all_node_attrs():
        if attr != 'node_mask':
            attrs[attr] = getattr(dense_graph, attr)[dense_graph.node_mask]
    #x = dense_graph.x[dense_graph.node_mask]
    ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=device), num_nodes.cumsum(dim=0)])
    batch = torch.repeat_interleave(torch.arange(len(num_nodes), device=device), num_nodes)

    # collect edge tensors
    edge_index, edge_attr = dense_to_sparse(
        adj =				dense_graph.edge_adjmat,
        cum_num_nodes_s	=	ptr,
        cum_num_nodes_t	=	ptr
    )
    
    for attr in dense_graph.get_all_edge_attrs():
        if attr != 'edge_mask' and attr != 'edge_adjmat':
            attrs[attr] = getattr(dense_graph, attr)[dense_graph.get_edge_mask_dense()]
            
    for attr in dense_graph.get_all_global_attrs():
        attrs[attr] = getattr(dense_graph, attr)

    # build sparse graph
    if batchify:
        cls = Batch
        kwargs = dict(_base_cls=SparseGraph)
    else:
        cls = SparseGraph
        kwargs = {}

    sparse_graph = cls(
        edge_index =	edge_index,
        edge_attr =		edge_attr,
        batch =			batch,
        ptr =			ptr,
        **attrs,
        **kwargs
    )

    if batchify:
        sparse_graph = build_graphs_batch(
            graph =				sparse_graph,
            batch =				batch,
            batch_size =		len(num_nodes),
            original_slice_dict = {},
            original_inc_dict = {},
            ptr =				ptr
        )
    

    return sparse_graph

def dense_edges_to_sparse_edges(
        dense_edges: DenseEdges,
        cum_num_nodes_s: IntTensor,
        cum_num_nodes_t: IntTensor
    ) -> SparseGraph:

    edge_index, edge_attr = dense_to_sparse(
        adj =				dense_edges.edge_adjmat,
        cum_num_nodes_s =	cum_num_nodes_s,
        cum_num_nodes_t =	cum_num_nodes_t
    )

    sparse_edges = SparseEdges(
        edge_index =	edge_index,
        edge_attr =		edge_attr
    )

    return sparse_edges


def get_node_mask_from_batch(
        batch: Tensor,
        batch_size: int=None,
        num_nodes: Tensor=None,
        max_num_nodes: int=None
    ) -> BoolTensor:
    """
    This method is taken from the PyTorch Geometric library:
    https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/utils/_to_dense_batch.py
    to perform only the batch to mask part
    """

    if batch_size is None:
        batch_size = int(batch.max()) + 1

    if num_nodes is None:
        num_nodes = scatter(batch.new_ones(batch.size(0)), batch, dim=0,
                        dim_size=batch_size, reduce='sum')
    cum_nodes = cumsum(num_nodes)

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())

    tmp = torch.arange(batch.size(0), device=batch.device) - cum_nodes[batch]
    idx = tmp + (batch * max_num_nodes)

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool,
                       device=batch.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)

    return mask


def get_batch_from_node_mask(node_mask: BoolTensor) -> Tensor:
    """Get the batch tensor from a node mask tensor

    Parameters
    ----------
    node_mask : BoolTensor
        boolean tensor of shape (batch_size, num_nodes) where True values
        indicate the presence of a node

    Returns
    -------
    Tensor
        tensor of shape (num_nodes,) containing the batch index for each node
    """
    return node_mask.nonzero(as_tuple=False)[:, 0]


def get_ptr_from_node_mask(node_mask: BoolTensor) -> Tensor:
    """Get the ptr tensor from a node mask tensor

    Parameters
    ----------
    node_mask : BoolTensor
        boolean tensor of shape (batch_size, num_nodes) where True values
        indicate the presence of a node

    Returns
    -------
    Tensor
        tensor of shape (batch_size+1,) containing the cumulative number of nodes
        for each example in the batch
    """
    return torch.cat([torch.zeros(1, device=node_mask.device, dtype=torch.long), node_mask.sum(dim=-1).cumsum(dim=0)])


def get_node_mask_from_num_nodes(num_nodes: IntTensor, max_num_nodes: int=None) -> BoolTensor:
    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())
    indices = torch.arange(max_num_nodes, device=num_nodes.device).unsqueeze(0).expand(num_nodes.size(0), -1)
    return indices < num_nodes.unsqueeze(-1)


def get_bipartite_edge_mask_dense(
        node_mask_a: Tensor,
        node_mask_b: Tensor
    ) -> Tensor:

    e_mask1 = node_mask_a.unsqueeze(-1)		# *, na, 1
    e_mask2 = node_mask_b.unsqueeze(-2)		# *, 1, nb
    e_mask = e_mask1 * e_mask2				# *, na, nb

    return e_mask

def get_edge_mask_dense(
        node_mask: Tensor=None,
        edge_mask: Tensor=None,
        remove_self_loops: bool = False,
        only_triangular: bool = False
    ) -> Tensor:
    if edge_mask is None:
        e_mask = get_bipartite_edge_mask_dense(node_mask, node_mask)
    else:
        e_mask = edge_mask
    
    if only_triangular:
        # remove upper triangular part and diagonal
        e_mask = torch.tril(e_mask, diagonal=-1)

    elif remove_self_loops:
        # remove self loops
        e_mask = e_mask * (1 - torch.eye(e_mask.size(-1), device=e_mask.device)).unsqueeze(0)

    return e_mask


def get_node_edge_mask_dense(node_mask: Tensor) -> Tuple[Tensor, Tensor]:
    """build the node and edge masks from a squeezed node mask

    Parameters
    ----------
    node_mask : Tensor
        boolean mask of shape (*, num_nodes)

    Returns
    -------
    x_mask : Tensor
        boolean tensor of shape (*, num_nodes) to be applied on node features
        tensors
    e_mask : Tensor
        boolean tensor of shape (*, num_nodes, num_nodes) to be applied on
        edge features adjacency matrices
    """
    x_mask = node_mask						# bs, n
    e_mask = get_edge_mask_dense(node_mask)	# bs, n, n

    return x_mask, e_mask


################################################################################
#                                    CHECKS                                    #
################################################################################
def is_matrix_symmetric(mat: Tensor) -> bool:
    return torch.allclose(mat, torch.transpose(mat, 1, 2))