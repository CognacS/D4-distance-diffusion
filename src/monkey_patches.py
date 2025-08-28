from typing import Any, List, Optional
from typing_extensions import Self

# monkey patch torch.repeat_interleave to fix a bug for mps devices
import torch
repeat_interleave_original = torch.repeat_interleave

def repeat_interleave_patched(input, repeats, dim=None):
    if isinstance(repeats, torch.Tensor) and repeats.device == torch.device('mps:0'):
        if repeats.sum().item() == 0:
            final_shape = list(input.shape)
            final_shape[dim] = 0
            return torch.zeros(*final_shape, device=input.device, dtype=input.dtype)
    
    return repeat_interleave_original(input, repeats, dim=dim)

repeat_interleave_patched.__name__ = repeat_interleave_original.__name__
repeat_interleave_patched.__doc__ = repeat_interleave_original.__doc__
torch.repeat_interleave = repeat_interleave_patched
torch.Tensor.repeat_interleave

# monkey patch Batch.from_data_list to fix a bug for mps devices
import torch_geometric.data
from torch_geometric.data.data import BaseData

from_data_list_original = torch_geometric.data.Batch.from_data_list

@classmethod
def from_data_list_patched(
        cls,
        data_list: List[BaseData],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ) -> Self:
        r"""Constructs a :class:`~torch_geometric.data.Batch` object from a
        list of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects.
        The assignment vector :obj:`batch` is created on the fly.
        In addition, creates assignment vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`.
        """
        
        batch = from_data_list_original(
            data_list=data_list,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        if batch.x.device == torch.device('mps:0'):
            batch.batch = batch.batch.to(batch.x.device)
            batch.ptr = batch.ptr.to(batch.x.device)

        return batch

from_data_list_patched.__name__ = from_data_list_original.__name__
from_data_list_patched.__doc__ = from_data_list_original.__doc__
torch_geometric.data.Batch.from_data_list = from_data_list_patched