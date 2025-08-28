from typing import Dict

from torch import nn

class Metric(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, **kwargs) -> Dict:
        raise NotImplementedError