
import torch
from torch import Tensor
from torch_geometric.data import Data

from src.datatypes.features.posenc import SinusoidalPosEmb
from src.datatypes.features.core import Feature, convert_if_mismatch_type, unpack_graph
from src.datatypes.dense import DenseGraph
from src.datatypes.sparse import SparseGraph

from src.datatypes.features import reg_features
    

@reg_features.register('indegree')
class InDegreeFeature(Feature):

    def __init__(self, use_emb=True, dim=16, scale=1.0, **kwargs):
        super().__init__()
        self.emb = SinusoidalPosEmb(dim, scale=scale) if use_emb else lambda x: x.unsqueeze(-1)

    def get_added_dims(self):
        num_embs = self.emb.dim if isinstance(self.emb, SinusoidalPosEmb) else 1
        return {'x': num_embs}

    def __call__(self, graph: Data) -> Data:

        graphs = unpack_graph(graph)

        for graph_edges in graphs:
            graph, graph_outgoing_edges = graph_edges
            # in undirected graph, remv -> surv is the same as surv -> remv
            # so outdegree of surv is the same as indegree of remv
            if graph_outgoing_edges is None:
                add_indegree = 0
            else:
                add_indegree = graph_outgoing_edges.outdegree
                add_indegree = convert_if_mismatch_type(graph_outgoing_edges, graph, add_indegree)
            indegree = graph.indegree + add_indegree
            # embed if needed
            indegree = self.emb(indegree)
            # concatenate to the features
            graph.x = torch.cat([graph.x, indegree], dim=-1)

@reg_features.register('outdegree')
class OutDegreeFeature(Feature):

    def __init__(self, use_emb=True, dim=16, scale=1.0, **kwargs):
        super().__init__()
        self.emb = SinusoidalPosEmb(dim, scale=scale) if use_emb else lambda x: x.unsqueeze(-1)

    def get_added_dims(self):
        num_embs = self.emb.dim if isinstance(self.emb, SinusoidalPosEmb) else 1
        return {'x': num_embs}

    def __call__(self, graph: Data) -> Data:

        graphs = unpack_graph(graph)

        for graph_edges in graphs:
            graph, graph_outgoing_edges = graph_edges
            # in undirected graph, remv -> surv is the same as surv -> remv
            # so indegree of surv is the same as outdegree of remv
            if graph_outgoing_edges is None:
                add_outdegree = 0
            else:
                add_outdegree = graph_outgoing_edges.indegree
                add_outdegree = convert_if_mismatch_type(graph_outgoing_edges, graph, add_outdegree)
            outdegree = graph.outdegree + add_outdegree
            # embed if needed
            outdegree = self.emb(outdegree)
            # concatenate to the features
            graph.x = torch.cat([graph.x, outdegree], dim=-1)


@reg_features.register('nodes_num')
class NodesNumFeature(Feature):

    def __init__(self, use_emb=True, dim=16, scale=1.0, **kwargs):
        super().__init__()
        self.emb = SinusoidalPosEmb(dim, scale=scale) if use_emb else lambda x: x.unsqueeze(-1)

    def get_added_dims(self):
        num_embs = self.emb.dim if isinstance(self.emb, SinusoidalPosEmb) else 1
        return {'y': num_embs}

    def __call__(self, graph: Data) -> Data:
        if isinstance(graph, (SparseGraph, DenseGraph)):
            graph = graph
        elif isinstance(graph, (tuple, list)):
            graph = graph[0]

        nodes_num = graph.num_nodes_per_sample
        nodes_num = self.emb(nodes_num)
        if graph.y is None:
            graph.y = nodes_num
        else:
            graph.y = torch.cat([graph.y, nodes_num], dim=-1)

        return graph


@reg_features.register('depth')
class NodesDepthFeature(Feature):

    def __init__(self, dim=16, scale=1.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.emb = SinusoidalPosEmb(self.dim, scale=scale)

    def get_added_dims(self):
        return {'x': self.dim}

    def __call__(self, graph: Data) -> Data:

        if isinstance(graph, (SparseGraph, DenseGraph)):
            pass
        elif isinstance(graph, (tuple, list)):
            graph = graph[0]

        node_depth = self.emb(graph.node_depth)
        
        graph.x = torch.cat([graph.x, node_depth], dim=-1)



def main():
    emb = SinusoidalPosEmb(16)
    inp_y = torch.tensor([16, 2800])
    inp_x = torch.tensor([[17, 2], [1, 17]])

    print(emb(inp_y))
    print(emb(inp_x))

if __name__=='__main__':
    main()