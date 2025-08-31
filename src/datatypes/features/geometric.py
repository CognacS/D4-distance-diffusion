
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

        indegree = graph.indegree
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

        outdegree = graph.outdegree
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

        nodes_num = graph.num_nodes_per_sample
        nodes_num = self.emb(nodes_num)
        if graph.y is None:
            graph.y = nodes_num
        else:
            graph.y = torch.cat([graph.y, nodes_num], dim=-1)

        return graph