from __future__ import annotations
from typing import Any, List, Union, Dict

import copy

import torch
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.data import Data, Batch

from torch_geometric.utils import degree
from src.datatypes.utils import one_hot
from src.datatypes.core import Graph

################################################################################
#                          SPARSE GRAPH DATASTRUCTURE                          #
################################################################################

class SparseGraph(Graph):
    """This class is a wrapper for torch_geometric.data.Data, which is used to
    represent sparse graphs. It adds the ability to decide how to copy each attribute."""

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if Graph.is_global_attr(key):
            if value.ndim == 1 and len(value) == 1:
                return 0
            else:
                return None
        self.keys()

        return super().__cat_dim__(key, value, *args, **kwargs)


    def extract_attributes(self, attributes: List[str]) -> Dict[str, Tensor]:
        """Extract the specified attributes from the graph.

        Parameters
        ----------
        attributes : List[str]
            list of attribute names to be extracted

        Returns
        -------
        out : Dict[str, Tensor]
            dictionary of attributes
        """

        out = {}

        for key in attributes:
            out[key] = self[key]

        return out


    def selective_deepcopy(self, attributes: List[str]) -> 'SparseGraph':
        """Deepcopy the attributes of the graph, given
        by the list of attribute names.

        Parameters
        ----------
        attributes : List[str]
            list of attribute names to be copied

        Returns
        -------
        out : SparseGraph
            deepcopy of the graph with only the specified attributes
        """

        out = self.__class__.__new__(self.__class__)
        out.__init__()

        for key in attributes:
            out[key] = copy.deepcopy(self[key])

        return out
    

    def setattrs(self, **kwargs):
        """Set attributes of the graph."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def collapsed(self) -> bool:
        """Check if the graph is collapsed."""
        return self.x.ndim == 1 and self.edge_attr.ndim == 1
    
    @property
    def num_nodes_per_sample(self) -> int:
        """Get the number of nodes in the graph or batch of graphs."""
        # get number of nodes
        if hasattr(self, 'ptr'):
            num_nodes = self.ptr[1:] - self.ptr[:-1]
        else:
            num_nodes = self.num_nodes

        return num_nodes
    
    @property
    def outdegree(self) -> Tensor:
        """Get the outdegree of the graph or batch of graphs."""
        if hasattr(self, 'node_outdegree'):
            return self.node_outdegree
        return degree(self.edge_index[0], self.num_nodes)
    
    @property
    def indegree(self) -> Tensor:
        """Get the indegree of the graph or batch of graphs."""
        if hasattr(self, 'node_indegree'):
            return self.node_indegree
        return degree(self.edge_index[1], self.num_nodes)
    
    @property
    def num_nodes(self) -> int:
        """Get the number of nodes in the graph or batch of graphs."""
        if hasattr(self, 'global_num_nodes'):
            return self.global_num_nodes
        return self.x.shape[0]

    
    def collapse(self) -> SparseGraph:
        """returns a SparseGraph where each entry is a class instead of a feature
        vector

        Returns
        -------
        collapsed_graph : SparseGraph
            this graph but with classes instead of feature vectors
        """

        if not self.collapsed:
            # collapse to classes
            self.x =            torch.argmax(self.x, dim=-1)
            self.edge_attr =    torch.argmax(self.edge_attr, dim=-1)

        return self
    

    def to_onehot(self, num_classes_x: int, num_classes_e: int) -> SparseGraph:

        if self.collapsed:

            self.x =         one_hot(self.x,         num_classes = num_classes_x, dtype=torch.float)
            self.edge_attr = one_hot(self.edge_attr, num_classes = num_classes_e, dtype=torch.float)
    
        return self


def from_data(data: Data) -> SparseGraph:
    """Convert a torch_geometric.data.Data object to a SparseGraph object."""
    return SparseGraph(**data.to_dict())


def create_empty_graph(
        batch_size: int,
        initialization: Dict[str, Tensor],
        device=None
    ):
    # reverse initialization:
    # dict of arrays -> array of dicts
    keys = list(initialization.keys())
    rev_initilization = [
        {k: initialization[k][i] for k in keys}
        for i in range(batch_size)
    ]

    # make collection of graphs with empty nodes and edges
    out = [
        SparseGraph(
            x = 			torch.empty(0, dtype=torch.long, device=device),
            edge_index = 	torch.empty((2, 0), dtype=torch.long, device=device),
            edge_attr = 	torch.empty(0, dtype=torch.long, device=device),
            **init
        ) for init in rev_initilization
    ]

    # collate graphs
    out = Batch.from_data_list(out)

    return out

class SparseEdges(SparseGraph):
    """This class is a wrapper for torch_geometric.data.Data, which is used to
    represent sparse edges between two subgraphs."""

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'edge_index':
            return torch.tensor([[self.num_nodes_s], [self.num_nodes_t]])
        if 'num_nodes' in key:
            return 0

        return super().__inc__(key, value, *args, **kwargs)
    
    @property
    def collapsed(self) -> bool:
        """Check if the graph is collapsed."""
        return self.edge_attr.ndim == 1
    

    def transpose(self) -> SparseEdges:
        """Transpose the edge index."""
        self.edge_index = self.edge_index.flip(0)
        if hasattr(self, 'num_nodes_s'):
            # swap sizes
            self.num_nodes_s, self.num_nodes_t = self.num_nodes_t, self.num_nodes_s
        return self

    
    def collapse(self) -> SparseGraph:
        """returns a SparseGraph where each entry is a class instead of a feature
        vector

        Returns
        -------
        collapsed_graph : SparseGraph
            this graph but with classes instead of feature vectors
        """

        if not self.collapsed:
            self.edge_attr =    torch.argmax(self.edge_attr, dim=-1)

        return self
    

    def to_onehot(self, num_classes_e: int) -> SparseGraph:

        if self.collapsed:
            self.edge_attr = one_hot(self.edge_attr, num_classes = num_classes_e, dtype=torch.float)
    
        return self
    



def to_directed(edge_index: Tensor, edge_attr: Tensor, lower_to_higher: bool = True):

    # compare source and target nodes
    edge_mask = edge_index[0] <= edge_index[1] if lower_to_higher else edge_index[0] >= edge_index[1]

    # mask edges and remove duplicates
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask]

    return edge_index, edge_attr