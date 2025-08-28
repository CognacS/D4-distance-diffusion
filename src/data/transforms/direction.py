from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected

from src.datatypes.sparse import SparseGraph, to_directed

from src.data.transforms import reg_transforms

@reg_transforms.register('to_undirected')
class MyToUndirected(BaseTransform):
    r"""Note: this is a fixed version of the original ToUndirected transform
    without bugs on recognizing edge attributes. This is also specialized for
    SparseGraphs.
    Converts a homogeneous or heterogeneous graph to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge
    :math:`(i,j) \in \mathcal{E}` (functional name: :obj:`to_undirected`).
    In heterogeneous graphs, will add "reverse" connections for *all* existing
    edge types.

    Args:
        reduce (str, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"add"`)
        merge (bool, optional): If set to :obj:`False`, will create reverse
            edge types for connections pointing to the same source and target
            node type.
            If set to :obj:`True`, reverse edges will be merged into the
            original relation.
            This option only has effects in
            :class:`~torch_geometric.data.HeteroData` graph data.
            (default: :obj:`True`)
    """
    def __init__(self, reduce: str = "add", merge: bool = True, **kwargs):
        self.reduce = reduce
        self.merge = merge

    def forward(
        self,
        data: SparseGraph,
    ) -> SparseGraph:
        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            keys, values = [], []
            for key, value in store.items():
                if key == 'edge_index':
                    continue

                # here is the fix: use data instead of store
                # for recognizing edge attributes
                if data.is_edge_attr(key):
                    keys.append(key)
                    values.append(value)

            store.edge_index, values = to_undirected(
                store.edge_index, values, reduce=self.reduce)

            for key, value in zip(keys, values):
                store[key] = value

        return data


@reg_transforms.register('to_directed')
class MyToDirected(BaseTransform):
    r"""Converts a graph to a directed graph, removing
    one of the two edges between two nodes. Which one is
    removed is determined by the `lower_to_higher` parameter.

    Args:
        lower_to_higher (bool, optional): If set to :obj:`True`, will keep
            edges from lower to higher node indices. If set to :obj:`False`,
            will keep edges from higher to lower node indices. (default: :obj:`True`)
    """


    def __init__(self, lower_to_higher: bool = True, **kwargs):
        self.lower_to_higher = lower_to_higher


    def forward(
        self,
        data: SparseGraph,
    ) -> SparseGraph:
        
        data.edge_index, data.edge_attr = to_directed(
            data.edge_index, data.edge_attr,
            lower_to_higher=self.lower_to_higher
        )

        return data