from typing import Dict
from torch_geometric.transforms import BaseTransform

class TransformAdapter:

    def __init__(self, map: Dict=None):
        if map is None:
            self.map = {}
        else:
            self.map = map

    def instantiate(self, **kwargs) -> BaseTransform:
        raise NotImplementedError()