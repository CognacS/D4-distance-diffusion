from typing import List

from torch import nn

class DataCollection(nn.Module):
    """Torch module which collects data points as
    parameters and allows for easy access to them.
    An initial size is set, and new data can be added
    in batches through the `add_data` method.
    """
    def __init__(self, max_size: int):
        super().__init__()
        self.data = nn.ParameterList()
        self.max_size = max_size

    def add_data(self, data: List):

        remainder = self.max_size - len(self.data)

        if remainder > 0:
            if len(data) > remainder:
                self.data.extend(data[:remainder])
            else:
                self.data.extend(data)


    def clear(self):
        self.data = []

    def forward(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    