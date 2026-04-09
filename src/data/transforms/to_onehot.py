from typing import Dict

from torch_geometric.transforms import BaseTransform
from src.datatypes.sparse import SparseGraph
from src.data.transforms.core import TransformAdapter
from src.datatypes.utils import one_hot

from src.data.transforms import reg_transforms


class ToOneHotGraph(BaseTransform):
    def __init__(self, attrs_to_num_cls, **kwargs):
        super().__init__(**kwargs)
        self.attrs_to_num_cls = attrs_to_num_cls

    def forward(self, data: SparseGraph):
        return data.to_onehot(self.attrs_to_num_cls)

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__, self.attrs_to_num_cls
        )


@reg_transforms.register('to_onehot_graph')
class ToOneHotGraphAdapter(TransformAdapter):

    def instantiate(self, **kwargs) -> BaseTransform:
        data_resources = kwargs['data_resources']
        info = data_resources.info_total

        attrs_to_num_cls = {k: info[v] for k, v in self.map.items() if v in info}

        for key, value in info.items():
            if key.startswith('num_cls_node_'):
                attr_name = key.removeprefix('num_cls_')
                attrs_to_num_cls[attr_name] = value

        tr = ToOneHotGraph(
            attrs_to_num_cls
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

    def instantiate(self, **kwargs) -> BaseTransform:
        data_resources = kwargs['data_resources']
        info = data_resources.info_total

        tr = ToOneHot(
            attr_name=self.attr_name,
            num_classes=info[self.map['num_classes']]
        )

        return tr