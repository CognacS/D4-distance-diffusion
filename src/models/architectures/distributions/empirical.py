from typing import Optional, List, Union, Dict

import torch
from torch import Tensor
import torch.nn as nn
from src.models import reg_architectures

@reg_architectures.register()
class EmpiricalSampler(nn.Module):

    def __init__(
            self,
            dataset_info: Dict,
            device: Optional[str]=None,
            **kwargs
        ):
        super().__init__()

        self.output_type = 'regressor'

        # get the histogram of number of nodes
        nodes_hist = [(int(k), int(v)) for k, v in dataset_info['num_nodes_hist'].items()]
        nodes_hist = list(zip(*nodes_hist))
        nodes_idx = torch.tensor(nodes_hist[0], dtype=torch.int64, device=device)
        nodes_weights = torch.tensor(nodes_hist[1], dtype=torch.float, device=device)
        
        # e.g. histogram of the number of nodes in the dataset
        self.property_map = nn.Parameter(nodes_idx, requires_grad=False)
        self.property_histograms = nn.Parameter(nodes_weights, requires_grad=False)


    def forward(
            self,
            batch_size: Optional[int]=None,
            **kwargs
        ):

        # sample from the empirical distribution (multinomial)
        number_of_nodes_idx = torch.multinomial(
            input =         self.property_histograms,
            num_samples =   batch_size,
            replacement =   True
        )

        # retrieve the number of nodes from the property map
        number_of_nodes = self.property_map[number_of_nodes_idx]

        return number_of_nodes