
import torch
from torch import Tensor
from torch_geometric.data import Data

from src.datatypes.features.posenc import SinusoidalPosEmb
from src.datatypes.features.core import Feature, convert_if_mismatch_type, unpack_graph
from src.datatypes.dense import DenseGraph
from src.datatypes.sparse import SparseGraph
from src.datatypes.core import Graph

from src.datatypes.features import reg_features
    

@reg_features.register('concat_att')
class ConcatenateAttributeFeature(Feature):

    def __init__(self, attribute_name, use_emb=False, dim=16, scale=1.0, **kwargs):
        super().__init__()
        self.attribute_name = attribute_name
        self.dim = dim
        self.emb = SinusoidalPosEmb(dim, scale=scale) if use_emb else lambda x: x

    def get_added_dims(self):
        num_embs = self.emb.dim if isinstance(self.emb, SinusoidalPosEmb) else self.dim
        if Graph.is_node_attr(self.attribute_name):
            return {'x': num_embs}
        elif Graph.is_edge_attr(self.attribute_name):
            return {'e': num_embs}
        elif Graph.is_global_attr(self.attribute_name):
            return {'y': num_embs}

    def __call__(self, graph: Data) -> Data:

        graphs = unpack_graph(graph)

        if Graph.is_node_attr(self.attribute_name):
            for graph, graph_outgoing_edges in graphs:
                attr = self.emb(graph[self.attribute_name]).float()
                assert attr.shape[-1] == self.dim
                graph.x = torch.cat([graph.x, attr], dim=-1)
            
        elif Graph.is_edge_attr(self.attribute_name):
            for graph, graph_outgoing_edges in graphs:
                
                if graph is not None and hasattr(graph, 'edge_attr'):
                    attr = self.emb(graph[self.attribute_name]).float()
                    assert attr.shape[-1] == self.dim
                    graph.edge_attr = torch.cat([graph.edge_attr, attr], dim=-1)
                
                if graph_outgoing_edges is not None:
                    attr = self.emb(graph_outgoing_edges[self.attribute_name]).float()
                    assert attr.shape[-1] == self.dim
                    graph_outgoing_edges.edge_attr = torch.cat([graph_outgoing_edges.edge_attr, attr], dim=-1)
        
        elif Graph.is_global_attr(self.attribute_name):
            attr = self.emb(graphs[0][self.attribute_name])
            assert attr.shape[-1] == self.dim
            graphs[0].y = torch.cat([graphs[0].y, attr], dim=-1)