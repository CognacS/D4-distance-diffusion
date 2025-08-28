from typing import List

from torch_geometric.data import Data

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