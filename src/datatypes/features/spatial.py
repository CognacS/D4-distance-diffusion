
import torch
from torch import Tensor
from torch_geometric.data import Data

from src.datatypes.features.posenc import SinusoidalPosEmb
from src.datatypes.features.core import Feature, convert_if_mismatch_type, unpack_graph
from src.datatypes.dense import DenseGraph
from src.datatypes.sparse import SparseGraph

from src.datatypes.features import reg_features
    

@reg_features.register('distance')
class DistanceFeature(Feature):

    def __init__(self, use_emb=True, dim=16, scale=1.0, **kwargs):
        super().__init__()
        self.emb = SinusoidalPosEmb(dim, scale=scale) if use_emb else lambda x: x.unsqueeze(-1)

    def get_added_dims(self):
        num_embs = self.emb.dim if isinstance(self.emb, SinusoidalPosEmb) else 1
        return {'e': num_embs}

    def __call__(self, graph: Data) -> Data:

        graphs = unpack_graph(graph)

        for graph_edges in graphs:
            graph, graph_outgoing_edges = graph_edges
            
            assert hasattr(graph, 'node_pos'), "Graph must have 'node_pos' attribute for distance calculation."
            
            node_pos = graph.node_pos
            idx = graph.edge_index
            
            distance = (node_pos[idx[0]] - node_pos[idx[1]]).norm(dim=-1)
            distance = self.emb(distance)
            
            # concatenate to the features
            graph.edge_attr = torch.cat([graph.edge_attr, distance], dim=-1)
            
@reg_features.register('cosine_position')
class CosinePositionFeature(Feature):
    
    def __init__(self, attribute_name='node_pos', use_emb=False, dim=16, **kwargs):
        super().__init__()
        self.attribute_name = attribute_name
        self.emb = SinusoidalPosEmb(dim) if use_emb else lambda x: x.unsqueeze(-1)

    def get_added_dims(self):
        num_embs = self.emb.dim if isinstance(self.emb, SinusoidalPosEmb) else 1
        return {'e': num_embs}

    def __call__(self, graph: Data) -> Data:

        graphs = unpack_graph(graph)

        for graph_edges in graphs:
            graph, graph_outgoing_edges = graph_edges
            
            assert hasattr(graph, self.attribute_name), f"Graph must have '{self.attribute_name}' attribute for cosine position calculation."
            node_pos = graph[self.attribute_name]
            # compute cosine of position vectors for each edge in edge_index
            node_pos_i = node_pos[graph.edge_index[0]]
            node_pos_j = node_pos[graph.edge_index[1]]
            norms = node_pos.norm(dim=-1)
            norms_i = norms[graph.edge_index[0]]
            norms_j = norms[graph.edge_index[1]]
            
            dots = (node_pos_i.unsqueeze(-2) @ node_pos_j.unsqueeze(-1)).squeeze(-1).squeeze(-1)
            cosine_position = dots / (norms_i * norms_j + 1e-8)
            cosine_position = self.emb(cosine_position)
            
            # concatenate to the features
            graph.edge_attr = torch.cat([graph.edge_attr, cosine_position.float()], dim=-1)