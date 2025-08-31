from typing import List

import torch
from torch_geometric.data import Data
from src.datatypes.utils import one_hot

DIM_X = 'dim_x'
DIM_E = 'dim_e'
DIM_Y = 'dim_y'


SPECIAL_PREFIX = 'special_'
GLOBAL_PREFIX = 'global_'
NODE_PREFIX = 'node_'
EDGE_PREFIX = 'edge_'

class Graph(Data):
    """This class gives structure to graph classes like SparseGraph, DenseGraph, etc."""

    def get_all_node_attrs(self) -> List[str]:
        """Get all attributes which are not node or edge attributes."""
        return [key for key in self.keys() if self.is_node_attr(key)]
    
    def get_all_edge_attrs(self) -> List[str]:
        """Get all attributes which are not node or edge attributes."""
        return [key for key in self.keys() if self.is_edge_attr(key)]
    
    def get_all_global_attrs(self) -> List[str]:
        """Get all attributes which are not node or edge attributes."""
        return [key for key in self.keys() if self.is_global_attr(key)]
    
    def get_all_special_attrs(self) -> List[str]:
        """Get all attributes which should have a special treatment."""
        return [key for key in self.keys() if self.is_special_attr(key)]

    @staticmethod
    def is_node_attr(key: str) -> bool:
        """Check if the attribute is a node attribute."""
        return key.startswith(NODE_PREFIX) or key == 'x' or key == 'batch'
    
    @staticmethod
    def is_edge_attr(key: str) -> bool:
        return key.startswith(EDGE_PREFIX)
    
    @staticmethod
    def is_global_attr(key: str) -> bool:
        return key.startswith(GLOBAL_PREFIX) or key == 'y'
    
    @staticmethod
    def is_special_attr(key: str) -> bool:
        return key.startswith(SPECIAL_PREFIX)
    
    @property
    def device(self):
        return getattr(self, self.keys()[0]).device
    
    
    def _collapse_attr(self, attr) -> 'Graph':
        if hasattr(self, attr) and getattr(self, attr) is not None:
            setattr(self, attr, torch.argmax(getattr(self, attr), dim=-1))
        return self


    def collapse(self, *attrs) -> 'Graph':
        """returns a DenseGraph where each entry is a class instead of a feature
        vector

        Returns
        -------
        collapsed_graph : DenseGraph
            this graph but with classes instead of feature vectors
        """

        if not self.collapsed:
            
            for attr in attrs:
                self._collapse_attr(attr)

            self.collapsed = True

        return self
    

    def _to_onehot_attr(self, attr, num_classes) -> 'Graph':
        if hasattr(self, attr) and getattr(self, attr) is not None:
            setattr(self, attr, one_hot(getattr(self, attr), num_classes = num_classes, dtype=torch.float))
        return self


    def to_onehot(self, attrs_to_num_cls=None, **kw_attrs_to_num_cls) -> 'Graph':
        
        if attrs_to_num_cls is None:
            attrs_to_num_cls = kw_attrs_to_num_cls
        attrs_to_num_cls = {**attrs_to_num_cls, **kw_attrs_to_num_cls}

        if self.collapsed:
            
            for attr, num_classes in attrs_to_num_cls.items():
                self._to_onehot_attr(attr, num_classes)

            self.collapsed = False
    
        return self