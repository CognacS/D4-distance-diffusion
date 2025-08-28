from typing import Union, Tuple, Dict, Optional

import torch
from torch import Tensor

from torch_geometric.nn import MLP
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.conv import (
    GINEConv,
    MessagePassing,
)


class GINE(BasicGNN):

    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

        return GINEConv(mlp, **kwargs)
    

from src.models import reg_architectures
from src.models.architectures.gnn.core import SupervisedGNN

@reg_architectures.register()
class GINEModel(SupervisedGNN):
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
        encoder = GINE(
            in_channels =   in_dim,
            out_channels =  encoder_out_channels,
            edge_dim =      input_dims['e'],
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
