import torch
from torch import Tensor
import torch.nn.functional as F

from src.datatypes.sparse import SparseGraph
from src.datatypes.dense import DenseGraph, DenseEdges

###########################  BULK OPERATION METHODS  ###########################

def to_onehot_all(*data, **classes_nums):

    ret_data = []

    for i, d in enumerate(data):
        if isinstance(d, tuple):
            k, d = d
            ret_d = F.one_hot(
                d.long(), num_classes = classes_nums[k]
            ).float()

        elif isinstance(d, DenseEdges):
            ret_d = d.to_onehot(
                num_classes_e =	classes_nums['e']
            )
        
        elif isinstance(d, (DenseGraph, SparseGraph)):
            ret_d = d.to_onehot(
                num_classes_x =	classes_nums['x'],
                num_classes_e =	classes_nums['e']
            )

        elif isinstance(d, Tensor):
            if d.dtype == torch.bool:
                ret_d = d.unsqueeze(-1)

        elif d is None:
            ret_d = None

        else:
            raise NotImplementedError(f'{i}-th data of type {type(d)} during to_onehot_all')
        
        ret_data.append(ret_d)

    return ret_data



def mask_all(*data, **masks):

    ret_data = []

    for i, d in enumerate(data):
        if isinstance(d, tuple):
            k, d = d
            ret_d = d * masks[k].unsqueeze(-1)
        
        elif isinstance(d, DenseGraph):
            ret_d = d.apply_mask()

        elif d is None:
            ret_d = None

        else:
            raise NotImplementedError(f'{i}-th data of type {type(d)} during mask_all')

        ret_data.append(ret_d)

    return ret_data


#################################  ASSERTIONS  #################################

def assert_is_onehot(*data):

    tensor_dims = {
        'xd': ('dense nodes', 3),
        'xs': ('sparse nodes', 2),
        'ed': ('dense edges', 4),
        'es': ('sparse edges', 2)
    }

    for i, d in enumerate(data):
        if isinstance(d, tuple):

            k: str
            d: Tensor
            k, d = d
            
            assert d.ndim == tensor_dims[k][1], \
                f'Expected {tensor_dims[k][0]} to be of dimension {tensor_dims[k][1]}, got {d.ndim}'

        elif isinstance(d, DenseGraph):
            assert not d.collapsed, \
                'Expected the dense graph to be onehot'
        
        elif isinstance(d, SparseGraph):
            assert_is_onehot(
                ('xs', d.x),
                ('es', d.edge_attr)
            )

        else:
            raise NotImplementedError(f'Expected {i}-th data to be of type tuple, DenseGraph or SparseGraph, got {type(d)}')