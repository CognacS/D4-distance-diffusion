import torch
from torch import nn

from src.models import reg_architectures

@reg_architectures.register()
class ConstantModel(nn.Module):

    def __init__(self, output_type, value=1., device=None, **kwargs):
        super().__init__()

        self.value = value
        self.output_type = output_type


    def forward(
            self,
            x,
            batch_size,
            **kwargs
        ):

        c = torch.full((batch_size,), self.value, device=x.device)
        if self.output_type == 'classifier':
            c = c.unsqueeze(-1)
        return c
