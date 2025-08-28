#################################################################################
#        EXACTLY THE CODE FROM TORCH_GEOMETRIC, BUT WITH CUSTOM COLLATER        #
#################################################################################

from typing import List, Optional, Sequence, Union, Callable, Any

import torch.utils.data

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.loader.dataloader import Collater


class ExtendedCollater(Collater):

    def __call__(self, batch: List[Any]):
        # collate as usual
        batch = super().__call__(batch)

        for key in batch.get_all_special_attrs():
            value = getattr(batch, key)

            if isinstance(value, list):
                if isinstance(value[0], BaseData): # SPECIAL CASE FOR NESTED BATCHES
                    setattr(batch, key, Batch.from_data_list(
                        value,
                        follow_batch=self.follow_batch,
                        exclude_keys=self.exclude_keys,
                    ))

        return batch




class DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        collate_fn (callable, optional): Merges a list of data objects to the
            mini-batch. (default: :func:`torch_geometric.loader.dataloader.Collater`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        collate_fn: Optional[Callable] = None,
        **kwargs,
    ):
        
        if collate_fn is None:
            collate_fn = ExtendedCollater
        
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=collate_fn(follow_batch, exclude_keys),
            **kwargs,
        )