from typing import Tuple, Optional, List, Dict, Union, Callable, Any

import torch
from torch import Tensor, LongTensor, BoolTensor
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import coalesce

from src.datatypes.sparse import SparseGraph, SparseEdges

from src.datatypes.batch import build_graphs_batch

##############################  SIMPLE SUBGRAPH  ###############################

def get_subgraph(graph: Data, boolean_mask: torch.BoolTensor) -> Data:

    # compute the subgraph induced by the boolean mask
    new_graph = graph.subgraph(subset=boolean_mask)

    # mask edges
    """graph.edge_index, graph.edge_attr = subgraph(
        subset =			boolean_mask,
        edge_index =		graph.edge_index,
        edge_attr =			graph.edge_attr,
        relabel_nodes =		True,
        num_nodes = 		graph.num_nodes
    )

    # mask nodes
    graph.x = graph.x[boolean_mask]"""

    # account for the batch structure if needed
    # i.e., for each node select the batch to which it belongs
    if hasattr(graph, 'batch') and graph.batch is not None:
        batch_size = new_graph.num_graphs

        # in this case, batch is already masked
        batch = new_graph.batch

        # setup batch attributes
        new_graph =  build_graphs_batch(
            graph =					new_graph,
            batch =					batch, 
            batch_size =			batch_size,
            original_slice_dict =	graph._slice_dict,
            original_inc_dict =		graph._inc_dict
        )

    return new_graph

#################################  SPLITTING  ##################################

def get_new_labels(
        node_mask: BoolTensor,
        neg_node_mask: BoolTensor
    ) -> LongTensor:
    """Returns a new labels vector for the nodes of a graph. Entries of the output
    where node_mask is True contain the new labels for the nodes in the first
    partition, while entries where neg_node_mask is True contain the new labels
    for the nodes in the second partition.

    Parameters
    ----------
    node_mask : BoolTensor
        boolean mask of size (num_nodes,) indicating a partition of some graph
    neg_node_mask : BoolTensor
        negation of node_mask, indicating the other partition of the graph

    Returns
    -------
    node_relabelling : LongTensor
        new labels vector for the nodes of the graph
    """

    node_relabelling = torch.zeros(
        node_mask.size(0), dtype=torch.long, device=node_mask.device
    )
    node_relabelling[node_mask] = torch.arange(node_mask.sum().item(), device=node_mask.device)
    node_relabelling[neg_node_mask] = torch.arange(neg_node_mask.sum().item(), device=node_mask.device)

    return node_relabelling


def get_subgraph_splits(
        graph: SparseGraph,
        boolean_mask: BoolTensor,
        relabel_nodes: bool=True
    ) -> Tuple[SparseGraph, SparseGraph, SparseEdges, SparseEdges]:
    """Splits graph into subgraphs A and B, and adj matrices AB and BA.
        partition A: nodes with boolean_mask True
        partition B: nodes with boolean_mask False
    or equivalently:
        boolean_mask		-> A
        neg_boolean_mask 	-> B

    Parameters
    ----------
    graph : SparseGraph
        full graph in its sparse representation
    boolean_mask : BoolTensor
        boolean mask of size (num_nodes,) indicating which nodes belong to subgraph A
    other_global_fields : Optional[List[str]], optional
        list of other global fields to be copied to the subgraphs, with the same behavior
        as y, by default None
    relabel_nodes : bool, optional
        if True, the nodes of the subgraphs are relabelled in order to have consecutive indices, by default True

    Returns
    -------
    graph_a : SparseGraph
        subgraph A, where nodes are picked where boolean_mask is True
    graph_b : SparseGraph
        subgraph B, where nodes are picked where boolean_mask is False
    edges_ab : Tuple[Tensor, Tensor]
        edge_index and edge_attr of edges from graph_a to graph_b
    edges_ba : Tuple[Tensor, Tensor]
        edge_index and edge_attr of edges from graph_b to graph_a
    """
    

    # get the mask for nodes in subgraph B
    neg_boolean_mask = ~boolean_mask

    # select edges of the two partitions
    mask_edges_a = boolean_mask[graph.edge_index]
    mask_edges_b = ~mask_edges_a

    # compute masks for selecting edges in the respective partition
    mask_edges_aa = mask_edges_a[0] & mask_edges_a[1] # edges going from A to A
    mask_edges_ab = mask_edges_a[0] & mask_edges_b[1] # edges going from A to B
    mask_edges_ba = mask_edges_b[0] & mask_edges_a[1] # edges going from B to A
    mask_edges_bb = mask_edges_b[0] & mask_edges_b[1] # edges going from B to B

    # collect all masks for each split
    mask_edges_splits = [
        mask_edges_aa,
        mask_edges_ab,
        mask_edges_ba,
        mask_edges_bb
    ]

    # extract edges for each partition from the list of all edges
    splits_edge_index = [graph.edge_index[:, m] for m in mask_edges_splits]
    splits_edge_attr = [graph.edge_attr[m] for m in mask_edges_splits]
    
    # relabel nodes if needed
    if relabel_nodes:
        # get the new labels for the nodes
        node_relabelling = get_new_labels(boolean_mask, neg_boolean_mask)

        # relabel nodes
        splits_edge_index = [node_relabelling[e] for e in splits_edge_index]
    

    # deepcopy global attributes for subgraph A
    graph_a = graph.selective_deepcopy(graph.get_all_global_attrs() + graph.get_all_special_attrs())
    graph_b = graph.selective_deepcopy(graph.get_all_global_attrs() + graph.get_all_special_attrs())

    # build subgraph A (those with boolean_mask=True)
    graph_a.setattrs(
        **{k: graph[k][boolean_mask] for k in graph.get_all_node_attrs()},
        edge_index =	splits_edge_index[0],
        edge_attr = 	splits_edge_attr[0]
    )

    # build subgraph B (those with boolean_mask=False)
    graph_b.setattrs(
        **{k: graph[k][neg_boolean_mask] for k in graph.get_all_node_attrs()},
        edge_index =	splits_edge_index[3],
        edge_attr = 	splits_edge_attr[3]
    )


    # account for the batch structure if needed
    # i.e., for each node select the batch to which it belongs
    if hasattr(graph, 'ptr') and graph.ptr is not None:
        batch_size = graph.num_graphs

        # mask batch indices
        #batch_a = graph.batch[boolean_mask]
        #batch_b = graph.batch[neg_boolean_mask]

        # reuse the same function for building the batch attributes
        setup_batch = lambda g, b : build_graphs_batch(
            graph =					g,
            batch =					b,
            batch_size =			batch_size,
            original_slice_dict =	graph._slice_dict,
            original_inc_dict =		graph._inc_dict
        )

        # fill out graph_a and graph_b batch attributes
        graph_a = setup_batch(graph_a, graph_a.batch)
        graph_b = setup_batch(graph_b, graph_b.batch)


    edges_ab = SparseEdges(
        edge_index = splits_edge_index[1],
        edge_attr = splits_edge_attr[1],
        num_nodes_s = graph_a.num_nodes_per_sample,
        num_nodes_t = graph_b.num_nodes_per_sample,
        num_nodes = graph.num_nodes_per_sample
    )

    edges_ba = SparseEdges(
        edge_index = splits_edge_index[2],
        edge_attr = splits_edge_attr[2],
        num_nodes_s = graph_b.num_nodes_per_sample,
        num_nodes_t = graph_a.num_nodes_per_sample,
        num_nodes = graph.num_nodes_per_sample
    )


    return graph_a, graph_b, edges_ab, edges_ba

##################################  MERGING  ###################################

def relabel_edge_index(perm: LongTensor, edge_index: LongTensor) -> LongTensor:
    relabelling = torch.empty_like(perm)
    relabelling[perm] = torch.arange(perm.size(0), device=perm.device)

    # apply the relabelling to the node indices
    relabeled_edge_index = relabelling[edge_index]

    return relabeled_edge_index

def merge_subgraphs(
        graph_a: SparseGraph,
        graph_b: SparseGraph,
        edges_ab: SparseEdges,
        edges_ba: SparseEdges
    ) -> SparseGraph:
    """Returns a new graph that is the merge of two subgraphs, linked by edges_ab and edges_ba.

    Parameters
    ----------
    graph_a : SparseGraph
        first graph
    graph_b : SparseGraph
        second graph
    edges_ab : SparseEdges
        edges from graph_a to graph_b
    edges_ba : SparseEdges
        edges from graph_b to graph_a

    Returns
    -------
    graph : SparseGraph
        merged graph
    """

    #############################  VALUES MERGING  #############################
    # merge node features
    node_attrs = {}

    for f in graph_a.get_all_node_attrs():
        if hasattr(graph_a, f) and hasattr(graph_b, f):
            node_attrs[f] = torch.cat([graph_a[f], graph_b[f]], dim=0)

    # clone edge indices to avoid modifying the original ones
    edge_index_ab = edges_ab.edge_index.clone()
    edge_index_ba = edges_ba.edge_index.clone()

    # offset node indices of graph_b by the number of nodes in graph_a
    edge_index_ab[1] = edge_index_ab[1] + graph_a.num_nodes
    edge_index_ba[0] = edge_index_ba[0] + graph_a.num_nodes
    graph_b_edge_index = graph_b.edge_index + graph_a.num_nodes

    # merge edge indices
    edge_index = torch.cat([
        graph_a.edge_index,
        edge_index_ab,
        edge_index_ba,
        graph_b_edge_index
    ], dim=1)

    edge_index = merge_edge_indices(
        graph_a.edge_index,
        graph_b.edge_index,
        edges_ab.edge_index,
        edges_ba.edge_index,
        graph_a.num_nodes
    )

    # merge edge attributes
    edge_attr = torch.cat([
        graph_a.edge_attr,
        edges_ab.edge_attr,
        edges_ba.edge_attr,
        graph_b.edge_attr
    ], dim=0)


    ###############################  REORDERING  ###############################

    # batch bookkeeping: reorder the two graphs batch indices in order to be
    # compliant with the Batch class of torch_geometric

    # order the batch indices
    batch = node_attrs['batch']
    perm = torch.argsort(batch)

    # reorder nodes and batch indices
    for f, attr in node_attrs.items():
        node_attrs[f] = attr[perm]

    # x = x[perm]
    # batch = batch[perm]


    # apply the relabelling to the node indices
    relabeled_edge_index = relabel_edge_index(perm, edge_index)
    # reorder the edge indices
    reordered_edge_index, reordered_edge_attr = coalesce(relabeled_edge_index, edge_attr, batch.shape[0])

    
    ###############################  BUILD GRAPH  ##############################

    # copy one of the graphs to create the new one, taking global attrs
    graph = graph_a.selective_deepcopy(graph_a.get_all_global_attrs() + graph_a.get_all_special_attrs())

    # setup node and edge attributes
    graph.setattrs(
        **node_attrs,
        edge_index =	reordered_edge_index,
        edge_attr =		reordered_edge_attr
    )

    # setup batch attributes
    graph = build_graphs_batch(
        graph =					graph,
        batch =					node_attrs['batch'],
        batch_size =			graph_a.num_graphs,
        original_slice_dict =	graph_a._slice_dict,
        original_inc_dict =		graph_a._inc_dict
    )

    return graph


def merge_edge_indices(
        edge_index_aa: Tensor,
        edge_index_bb: Tensor,
        edge_index_ab: Tensor,
        edge_index_ba: Tensor,
        num_nodes_a: int
    ) -> Tensor:
    """Merges the edge indices of two graphs, given the edge indices between the two graphs."""

    # clone edge indices to avoid modifying the original ones
    edge_index_ab = edge_index_ab.clone()
    edge_index_ba = edge_index_ba.clone()

    # offset node indices of graph_b by the number of nodes in graph_a
    edge_index_ab[1] = edge_index_ab[1] + num_nodes_a
    edge_index_ba[0] = edge_index_ba[0] + num_nodes_a
    edge_index_bb = edge_index_bb + num_nodes_a

    # merge edge indices
    edge_index = torch.cat([
        edge_index_aa,
        edge_index_ab,
        edge_index_ba,
        edge_index_bb
    ], dim=1)

    return edge_index




def mask_dead_samples(curr_batch: SparseGraph, surv_batch: SparseGraph, remv_batch: SparseGraph):
    alive_samples = remv_batch.global_t > 0

    num_nodes = surv_batch.num_nodes_per_sample

    # mask entries
    alive_num_nodes = num_nodes[alive_samples]
    alive_ptr = F.pad(alive_num_nodes.cumsum(dim=0), (1, 0), value=0)



def merge_batches(batch_list: List[SparseGraph], alive_masks: List[BoolTensor]) -> SparseGraph:
    
    num_batches = len(batch_list)
    first_elem = batch_list[0]
    device = first_elem.x.device

    nodes_list = {k: [None] * num_batches for k in first_elem.get_all_node_attrs() if k != 'batch'}
    edge_index_list = [None] * num_batches
    edge_attr_list = [None] * num_batches
    globals_list = {k: [] * num_batches for k in first_elem.get_all_global_attrs()}
    b_list = [None] * num_batches
    ptr_list = [None] * num_batches

    cum_nodes = 0
    cum_bs = 0

    for i, (batch, mask) in enumerate(zip(batch_list, alive_masks)):
        for k in batch.get_all_node_attrs():
            if k != 'batch':
                nodes_list[k][i] = batch[k]
        edge_index_list[i] = batch.edge_index + cum_nodes
        edge_attr_list[i] = batch.edge_attr
        
        alive_num_nodes = batch.num_nodes_per_sample[mask]
        curr_bs = alive_num_nodes.shape[0]
        b_alive = torch.repeat_interleave(
            torch.arange(curr_bs, device=device),
            repeats=alive_num_nodes,
            dim=0
        )

        b_list[i] = b_alive + cum_bs
        ptr_list[i] = alive_num_nodes.cumsum(dim=0) + cum_nodes
        cum_nodes += batch.x.shape[0]
        cum_bs += curr_bs

        for k in batch.get_all_global_attrs():
            globals_list[k].append(batch[k][mask])


    merged_batch = SparseGraph(
        **{k: torch.cat(nodes_list[k], dim=0) for k in nodes_list},
        edge_index = torch.cat(edge_index_list, dim=1),
        edge_attr = torch.cat(edge_attr_list, dim=0),
        batch = torch.cat(b_list, dim=0),
        ptr = F.pad(torch.cat(ptr_list, dim=0), (1, 0), value=0),
        num_graphs = cum_bs,
        **{k: torch.cat(globals_list[k], dim=0) for k in globals_list}
    ) 

    return merged_batch



def merge_batches_2(batch_list: List[SparseGraph], alive_masks: List[BoolTensor]) -> SparseGraph:
    
    num_batches = len(batch_list)
    first_elem = batch_list[0]
    device = first_elem.x.device

    x_list = [None] * num_batches
    edge_index_list = [None] * num_batches
    edge_attr_list = [None] * num_batches
    globals_list = {k: [] * num_batches for k in first_elem.get_all_global_attrs()}
    b_list = [None] * num_batches
    ptr_list = [None] * num_batches

    cum_nodes = 0
    cum_bs = 0

    for i, (batch, mask) in enumerate(zip(batch_list, alive_masks)):
        x_list[i] = batch.x
        edge_index_list[i] = batch.edge_index + cum_nodes
        edge_attr_list[i] = batch.edge_attr
        
        alive_num_nodes = batch.num_nodes_per_sample[mask]
        curr_bs = alive_num_nodes.shape[0]
        b_alive = torch.repeat_interleave(
            torch.arange(curr_bs, device=device),
            repeats=alive_num_nodes,
            dim=0
        )

        b_list[i] = b_alive + cum_bs
        ptr_list[i] = alive_num_nodes.cumsum(dim=0) + cum_nodes
        cum_nodes += batch.x.shape[0]
        cum_bs += curr_bs

        for k in batch.get_all_global_attrs():
            globals_list[k].append(batch[k][mask])

    merged_batch = SparseGraph(
        x = torch.cat(x_list, dim=0),
        edge_index = torch.cat(edge_index_list, dim=1),
        edge_attr = torch.cat(edge_attr_list, dim=0),
        batch = torch.cat(b_list, dim=0),
        ptr = F.pad(torch.cat(ptr_list, dim=0), (1, 0), value=0),
        num_graphs = cum_bs,
        **{k: torch.cat(globals_list[k], dim=0) for k in globals_list}
    )
    

    return merged_batch

def merge_edge_batches(batch_list: List[SparseEdges], alive_masks: List[BoolTensor]) -> SparseEdges:
    
    num_batches = len(batch_list)

    edge_index_list = [None] * num_batches
    edge_attr_list = [None] * num_batches
    num_nodes_s_list = [None] * num_batches
    num_nodes_t_list = [None] * num_batches

    cum_nodes = torch.zeros(2, 1, dtype=torch.int32, device=batch_list[0].edge_index.device)

    for i, (batch, mask) in enumerate(zip(batch_list, alive_masks)):
        edge_index_list[i] = batch.edge_index + cum_nodes
        edge_attr_list[i] = batch.edge_attr
        num_nodes_s_list[i] = batch.num_nodes_s[mask]
        num_nodes_t_list[i] = batch.num_nodes_t[mask]
        cum_nodes[0] += batch.num_nodes_s.sum().item()
        cum_nodes[1] += batch.num_nodes_t.sum().item()


    merged_batch = SparseEdges(
        edge_index = torch.cat(edge_index_list, dim=1),
        edge_attr = torch.cat(edge_attr_list, dim=0),
        num_nodes_s = torch.cat(num_nodes_s_list, dim=0),
        num_nodes_t = torch.cat(num_nodes_t_list, dim=0),
    )

    return merged_batch