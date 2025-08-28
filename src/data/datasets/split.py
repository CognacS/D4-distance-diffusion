from typing import List, Dict

import os.path as osp

from torch.utils.data import Dataset

import random


def split_dataset(dataset: Dataset|List[Dataset], splits_and_indices: Dict[str, List[int]]) -> Dict[str, Dataset]:
    """Split a dataset or a list of datasets into multiple datasets according to the splits_and_indices dictionary.
    For example, if splits_and_indices is {'train': [0, 1, 2], 'test': [3, 4, 5]}, the function will return a list
    of two datasets, the first one containing the first three elements of the input dataset, and the second one
    containing the last three elements.
    """

    if isinstance(dataset, Dataset):
        dataset = [dataset]

    # split datasets
    split_datasets = {k: [] for k in splits_and_indices.keys()}


    for ds in dataset:
        for split_name, indices in splits_and_indices.items():
            split_datasets[split_name].append(ds.subset_from(indices, split_name))

    return split_datasets


def random_split_dataset(dataset: Dataset|List[Dataset], splits_and_fracts_or_nums: Dict[str, float]|Dict[str, int], seed: int = None) -> Dict[str, Dataset]:
    """Split a dataset or a list of datasets into multiple datasets according to the splits list. For example, if splits is [0.7, 0.3],
    the function will return a list of two datasets, the first one containing 70% of the elements of the input dataset,
    and the second one containing the remaining 30%. If splits is [3, 1], the function will return a list of two
    datasets, the first one containing the first three elements of the input dataset, and the second one containing
    the last element. The seed parameter can be used to ensure reproducibility.
    """

    if isinstance(dataset, Dataset):
        dataset = [dataset]
    
    n = len(dataset[0])
    assert all(len(ds) == n for ds in dataset), 'All datasets must have the same length to split them together'

    indices = list(range(n))

    if seed is not None:
        random.seed(seed)

    random.shuffle(indices)

    splits_and_indices = {}
    start = 0
    accumulate_rest = None
    for split_name, split_fract_or_num in splits_and_fracts_or_nums.items():
        # accumulate the rest of the data if needed
        if split_fract_or_num == None:
            if accumulate_rest is not None:
                raise ValueError('Only one split can have None as value to accumulate the rest of the data')
            accumulate_rest = split_name
            continue

        # split the data
        split_len = split_fract_or_num if isinstance(split_fract_or_num, int) else int(split_fract_or_num * n)
        end = start + split_len
        splits_and_indices[split_name] = indices[start:end]
        start = end

    # accumulate the rest of the data if needed
    if accumulate_rest is not None:
        splits_and_indices[accumulate_rest] = indices[start:]

    return split_dataset(dataset, splits_and_indices)
