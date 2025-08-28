from typing import Union, Tuple, Dict, Optional

import torch
from torch import nn, Tensor
    

from src.models import reg_architectures
from src.models.architectures.gnn.core import SupervisedGNN


class NoGNN(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: Union[Tensor, Tuple[Tensor, Tensor]], **kwargs):
        return torch.empty(x.shape[0], 0, device=x.device)
                


@reg_architectures.register()
class MLPModel(SupervisedGNN):
    def __init__(
            self,
            input_dims: Dict,
            output_type: str,
            ffn_config: Optional[Dict]=None,
            **kwargs
        ):

        # initialize a non-encoder
        # this will suppress nodes
        encoder = NoGNN()

        if ffn_config is None:
            ffn_config = {}

        # initialize the rest of the model
        super().__init__(
            encoder = encoder,
            encoder_out_channels = 0,
            output_type = output_type,
            globals_dim = input_dims['y'],
            use_all_layers = False,
            **ffn_config
        )



# def main():
#     m = MLPModel(
#         {'y': 3},
#         'classifier',
#         dict(
#             ffn_hidden_dim = 5,
#             ffn_num_layers = 2,
#             ffn_out_dim = 2,
#             aggregator_fn = 'mean'
#         )
#     )
#     print(m)

#     x = torch.ones(10, 2)
#     edge_index = torch.tensor([[0, 0, 1, 1], [1, 2, 3, 4]])
#     edge_attr = torch.ones(4, 2)
#     batch = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
#     y = torch.tensor([[1, 3, 4], [2, 1, 0]])

#     print(m(x, edge_index, edge_attr, y, batch, batch_size=2))

# if __name__=='__main__':
#     main()