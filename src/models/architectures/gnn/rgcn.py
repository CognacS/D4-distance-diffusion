from typing import Union, Tuple, Dict, Optional

import torch
from torch import Tensor

from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.conv import (
    RGCNConv,
    MessagePassing,
)


class RGCNConvLayer(RGCNConv):
    """This class is simply an adapter of RGCNConv to use it
    as a layer in a BasicGNN model, where only edge_weight
    and edge_attr can be passed. Then, edge_type from RGCNConv
    is passed as edge_attr to the super class.
    """

    def forward(self, x: Union[Tensor, Tuple[Tensor, Tensor]],
            edge_index: Tensor, edge_attr: Tensor = None):
        
        # if ndim == 2, then edge_attr is a one-hot encoding
        if edge_attr.ndim == 2:
            edge_type = torch.argmax(edge_attr, dim=-1)
        else:
            edge_type = edge_attr
        
        return super().forward(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type
        )


class RGCN(BasicGNN):

    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:

        return RGCNConvLayer(in_channels, out_channels, **kwargs)
    

from src.models import reg_architectures
from src.models.architectures.gnn.core import SupervisedGNN

@reg_architectures.register()
class RGCNModel(SupervisedGNN):
    def __init__(
            self,
            input_dims: Dict,
            encoder_out_channels: int,
            output_type: str,
            gnn_encoder_config: Dict,
            ffn_config: Optional[Dict]=None,
            use_all_layers: bool=False,
            **kwargs
        ):

        in_dim = input_dims['x']
        if output_type == 'encoder':
            in_dim += input_dims['y']

        # initialize encoder
        encoder = RGCN(
            in_channels =   in_dim,
            out_channels =  encoder_out_channels,
            num_relations = input_dims['e'],
            **gnn_encoder_config
        )

        if ffn_config is None:
            ffn_config = {}

        # initialize the rest of the model
        super().__init__(
            encoder = encoder,
            encoder_out_channels = encoder_out_channels,
            output_type = output_type,
            globals_dim = input_dims['y'],
            use_all_layers = use_all_layers,
            **ffn_config
        )
