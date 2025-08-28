from typing import Dict

from torch_geometric.transforms import BaseTransform
from src.datatypes.sparse import SparseGraph
from src.data.transforms.core import TransformAdapter
from src.data.datasets.core import DataResources

from src.datatypes.utils import one_hot

from src.data.transforms import reg_transforms


class ToOneHotGraph(BaseTransform):
    def __init__(self, num_classes_node, num_classes_edge, **kwargs):
        super().__init__(**kwargs)
        self.num_classes_node = num_classes_node
        self.num_classes_edge = num_classes_edge

    def forward(self, data: SparseGraph):
        return data.to_onehot(self.num_classes_node, self.num_classes_edge)

    def __repr__(self):
        return '{}(num_classes_node={}, num_classes_edge={})'.format(
            self.__class__.__name__, self.num_classes_node, self.num_classes_edge
        )


@reg_transforms.register('to_onehot_graph')
class ToOneHotGraphAdapter(TransformAdapter):

    def instantiate(self, data_resources: DataResources, **kwargs) -> BaseTransform:
        info = data_resources.info_total

        tr = ToOneHotGraph(
            num_classes_node=info[self.map['num_classes_node']],
            num_classes_edge=info[self.map['num_classes_edge']]
        )

        return tr


class ToOneHot(BaseTransform):
    def __init__(self, attr_name, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.attr_name = attr_name
        self.num_classes = num_classes

    def forward(self, data: SparseGraph):
        data = data.clone()
        new_val = one_hot(getattr(data, self.attr_name), self.num_classes)
        setattr(data, self.attr_name, new_val)
        return data

    def __repr__(self):
        return '{}(attr_name={}, num_classes={})'.format(
            self.__class__.__name__, self.attr_name, self.num_classes
        )

    
@reg_transforms.register('to_onehot')
class ToOneHotAdapter(TransformAdapter):
    
    def __init__(self, attr_name: str, map: Dict[str, str], **kwargs):
        super().__init__(map, **kwargs)
        self.attr_name = attr_name

    def instantiate(self, data_resources: DataResources, **kwargs) -> BaseTransform:
        info = data_resources.info_total

        tr = ToOneHot(
            attr_name=self.attr_name,
            num_classes=info[self.map['num_classes']]
        )

        return tr