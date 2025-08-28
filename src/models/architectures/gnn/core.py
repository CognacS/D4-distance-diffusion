from typing import Optional, Dict, List, Callable

import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric.nn as gnn


from torch_geometric.nn.models.basic_gnn import BasicGNN


OUT_TYPE_ENCODER = 'encoder'
OUT_TYPE_CLASSIFIER = 'classifier'
OUT_TYPE_REGRESSOR = 'regressor'


def aggregate_list(
        x: Tensor|List[Tensor],
        batch: Tensor,
        batch_size: int,
        aggregate_fun: Callable[[Tensor, Tensor], Tensor],
        is_list: bool=False
    ) -> Tensor:
    """ Aggregate a list of tensors using the given function. """
    if is_list:
        x_cat = torch.cat(x, dim=0)
        batch_cat = torch.repeat_interleave(batch, repeats=batch_size)
        return aggregate_fun(x_cat, batch_cat, batch_size)
    else:
        return aggregate_fun(x, batch, batch_size)




class SupervisedGNN(nn.Module):
    def __init__(
            self,
            encoder: BasicGNN,
            encoder_out_channels: int,
            output_type: str,
            globals_dim: Optional[int]=None,
            use_all_layers: bool=False,
            ffn_hidden_dim: Optional[int]=None,
            ffn_num_layers: Optional[int]=None,
            ffn_out_dim: Optional[int]=None,
            aggregator_fn: Optional[str]='none'
        ):

        super().__init__()

        self.use_all_layers = use_all_layers
        self.encoder = encoder
        self.ffn_out_dim = ffn_out_dim

        if use_all_layers:
            # hack into gnn encoder to return all layers
            self.encoder.jk = lambda xs: xs
        
        self.output_type = output_type

        if output_type == OUT_TYPE_ENCODER:
            self.aggregator = None
            self.out_layers = nn.Identity()

        if output_type == OUT_TYPE_CLASSIFIER or output_type == OUT_TYPE_REGRESSOR:

            if aggregator_fn == 'add':
                self.aggregator = gnn.global_add_pool
            elif aggregator_fn == 'mean':
                self.aggregator = gnn.global_mean_pool
            elif aggregator_fn == 'max':
                self.aggregator = gnn.global_max_pool
            elif aggregator_fn == 'none':
                self.aggregator = None
            else:
                raise ValueError(f"Aggregator function {aggregator_fn} not supported")

            # prepare layers dimensions
            if ffn_num_layers > 1:
                dims = [
                    (encoder_out_channels + globals_dim, ffn_hidden_dim),
                    *[(ffn_hidden_dim, ffn_hidden_dim) for _ in range(ffn_num_layers - 2)],
                    (ffn_hidden_dim, ffn_out_dim)
                ]
            else:
                dims = [(encoder_out_channels + globals_dim, ffn_out_dim)]

            self.out_layers = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Dropout(p=0.1),
                        nn.Linear(in_dim, out_dim),
                        nn.ReLU()
                    )
                    for in_dim, out_dim in dims[:-1]
                ],
                nn.Linear(*dims[-1])
            )



    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Tensor,
            y: Optional[Tensor]=None,
            batch: Optional[Tensor]=None,
            batch_size: Optional[int]=None,
            num_nodes: Optional[Tensor]=None
        ):

        # concatenate the global features to the node features
        # to inject global information into the encoding process
        if y is not None and self.output_type == OUT_TYPE_ENCODER:

            # repeat y for each node in the examples
            y = torch.repeat_interleave(y, num_nodes, dim=0)
            x = torch.cat([x, y], dim=-1)

        # encode the graph
        x = self.encoder(
            x =				x,
            edge_index =	edge_index,
            edge_attr =		edge_attr
        )

        # aggregate the node features if needed
        if self.aggregator is not None:
            x = aggregate_list(x, batch, batch_size, self.aggregator, self.use_all_layers)

        # return the encoding if this is an encoder
        if self.output_type == OUT_TYPE_ENCODER:
            return x

        # concatenate global properties if needed
        if y is not None:
            x = torch.cat((x, y), dim=-1)
        
        # compute the output properties
        out = self.out_layers(x)

        # remove one dimension if the output is a single value
        if out.shape[-1] == 1:
            out = out.squeeze(-1)

        return out

        
            