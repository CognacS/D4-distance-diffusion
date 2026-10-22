from typing import List, Dict

import os.path as osp

from torch.utils.data import Dataset

import random
import math


def get_split_sizes(size, splits):
    """Resolve counts before writing any subsets; reject over-allocation."""
    sizes = {}
    remainder = None
    for name, value in splits.items():
        if value is None:
            if remainder is not None:
                raise ValueError('Only one split can have None as value to accumulate the rest of the data')
            remainder = name
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'Invalid size for split {name}: {value!r}')
        if value < 0 or not math.isfinite(value) or (isinstance(value, float) and value > 1):
            raise ValueError(f'Invalid size for split {name}: {value!r}')
        sizes[name] = value if isinstance(value, int) else int(value * size)
    allocated = sum(sizes.values())
    if allocated > size:
        raise ValueError(f'Splits request {allocated} records, but the dataset contains only {size}. No subsets were written.')
    if remainder is not None:
        sizes[remainder] = size - allocated
    elif allocated != size:
        if splits and all(isinstance(value, float) for value in splits.values()) and math.isclose(sum(splits.values()), 1.0):
            # Integer truncation may leave a few records for valid ratios.
            sizes[next(reversed(sizes))] += size - allocated
        else:
            raise ValueError(f'Splits cover {allocated} of {size} records. Set one split to None to include the remainder.')
    return sizes


def validate_cached_splits(datasets, splits, total_size):
    """Existing files must match the requested graph and SMILES counts."""
    for name, expected in get_split_sizes(total_size, splits).items():
        actual = [len(dataset) for dataset in datasets[name]]
        if any(count != expected for count in actual):
            raise ValueError(
                f'Cached split {name!r} has graph/SMILES counts {actual}; expected {expected}. '
                'The cached partition does not match the configuration. Audit and repair the splits before training.'
            )


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


def random_split_dataset(dataset: Dataset|List[Dataset], splits_and_fracts_or_nums: Dict[str, float]|Dict[str, int], seed: int = 0) -> Dict[str, Dataset]:
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

    split_sizes = get_split_sizes(n, splits_and_fracts_or_nums)
    random.Random(seed).shuffle(indices)

    splits_and_indices = {}
    start = 0
    for split_name, split_len in split_sizes.items():
        end = start + split_len
        splits_and_indices[split_name] = indices[start:end]
        start = end

    return split_dataset(dataset, splits_and_indices)
