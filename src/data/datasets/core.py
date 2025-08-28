from typing import List, Tuple, Union, Callable, Any, Optional, Type, Dict

from abc import ABC, abstractmethod
import os
import os.path as osp
from torch.utils.data import Dataset

from torch_geometric.data import InMemoryDataset
from torch_geometric.data.dataset import files_exist, to_list
from torch_geometric.io.fs import makedirs

import src.data.utils.storing as stutils


DEFAULT_DATASET_PATH = 'datasets'
DEFAULT_SPLITS = {
    'train': 0.8,
    'val': 0.2,
    'test': 0.2,
}

class DatasetException(Exception):
    pass




class RawDataset(Dataset, ABC):
    """Base class for raw datasets. This has the same functionalities
    as the torch_geometric Dataset on the raw/download part, altough simplified.
    This is useful for pre-looking at data, and to create a better pipeline."""

    def __init__(self, root: str, split: Optional[str] = None, pre_transform=None, pre_filter=None):
        super().__init__()

        self.root = root
        self.split = split

        if split is not None and not files_exist(self.raw_paths):
            raise DatasetException('Trying to instantiate a split of a dataset that was not split yet.')
        
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self._download()
    

    @property
    def root_split(self) -> str:
        if self.split is None:
            return self.root
        else:
            return osp.join(self.root, self.split)


    @property
    def other_file_names(self) -> List[str]:
        return []
    
    @property
    def other_paths(self) -> List[str]:
        return [osp.join(self.root_split, f) for f in self.other_file_names]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root_split, 'raw')
    
    @property
    def raw_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        files = self.raw_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [osp.join(self.raw_dir, f) for f in to_list(files)]
    
    def _download(self):
        if files_exist(self.raw_paths):  # pragma: no cover
            return

        makedirs(self.raw_dir)
        self.download()


    def save(self, data: Any, path: str):
        stutils.save_file(data, path)

    def load(self, path: str) -> Any:
        return stutils.load_file(path)
    
    def delete(self):
        if osp.exists(self.raw_dir):
            files_to_remove = self.raw_paths + self.other_paths
            for f in files_to_remove:
                if osp.exists(f):
                    os.remove(f)
            if len(os.listdir(self.raw_dir)) == 0:
                os.rmdir(self.raw_dir)
            if len(os.listdir(self.root_split)) == 0:
                os.rmdir(self.root_split)
            if len(os.listdir(self.root)) == 0:
                os.rmdir(self.root)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'
    
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        raise NotImplementedError

    @abstractmethod
    def download(self):
        raise NotImplementedError
    
    @abstractmethod
    def subset_from(self, dataset: Dataset, indices: List[int], name: str) -> Dataset:
        raise NotImplementedError
    

class ProcessedDataset(InMemoryDataset, ABC):

    def __init__(self, root: str, split: Optional[str] = None, transform=None, pre_transform=None, pre_filter=None):

        self.root = root
        self.split = split

        if split is not None and not files_exist(self.processed_paths):
            raise DatasetException('Trying to instantiate a split of a dataset that was not split yet.')
        
        super().__init__(self.root, transform, pre_transform, pre_filter)


    @property
    def root_split(self) -> str:
        if self.split is None:
            return self.root
        else:
            return osp.join(self.root, self.split)

    @property
    def other_file_names(self) -> List[str]:
        return ['pre_filter.pt', 'pre_transform.pt']
    
    @property
    def other_paths(self) -> List[str]:
        return [osp.join(self.root_split, f) for f in self.other_file_names]

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root_split, 'processed')
    

    def save_file(self, data: Any, path: str):
        stutils.save_file(data, path)

    def load_file(self, path: str) -> Any:
        return stutils.load_file(path)

    def delete(self):
        if osp.exists(self.processed_dir):
            files_to_remove = self.processed_paths + self.other_paths
            for f in files_to_remove:
                if osp.exists(f):
                    os.remove(f)
            if len(os.listdir(self.processed_dir)) == 0:
                os.rmdir(self.processed_dir)
            if len(os.listdir(self.root_split)) == 0:
                os.rmdir(self.root_split)
            if len(os.listdir(self.root)) == 0:
                os.rmdir(self.root)

    
    @abstractmethod
    def subset_from(self, dataset: Dataset, indices: List[int], name: str) -> Dataset:
        raise NotImplementedError




class DatasetWrapper:

    def __init__(
            self,
            wrappers_pairs: Optional[Type|Tuple[Type, Dict]|List[Type|Tuple[Type, Dict]]]=None
        ):
        if isinstance(wrappers_pairs, type):
            wrappers_pairs = (wrappers_pairs, {})
        if isinstance(wrappers_pairs, tuple):
            wrappers_pairs = [wrappers_pairs]
        
        self.wrappers_pairs = wrappers_pairs

    
    def __call__(
            self,
            dataset,
            transform=None
        ):
        if self.wrappers_pairs is None:
            return dataset
        
        for i, (w, w_kwargs) in enumerate(self.wrappers_pairs):
            if i == len(self.wrappers_pairs) - 1:
                dataset = w(dataset=dataset, transform=transform, **w_kwargs)
            else:
                dataset = w(dataset=dataset, **w_kwargs)
        
        return dataset
    


class DataResources(ABC):

    def __init__(self):
        self.dataset_wrapper = None

    def add_dataset_wrapper(self, dataset_wrappers: Optional[Type|Tuple[Type, Dict]|List[Type|Tuple[Type, Dict]]|DatasetWrapper]=None):
        if not isinstance(dataset_wrappers, DatasetWrapper):
            dataset_wrapper = DatasetWrapper(wrappers_pairs=dataset_wrappers)
        self.dataset_wrapper = dataset_wrapper
        return self

    def wrap_dataset(self, dataset, transform=None):
        if self.dataset_wrapper is not None:
            return self.dataset_wrapper(dataset, transform=transform)
        
        dataset.transform = transform
        return dataset

    def transforms_to_pipeline(self, transforms):
        return transforms_to_pipeline(transforms=transforms, data_resources=self)

    @abstractmethod
    def prepare_data(self):
        raise NotImplementedError

    @abstractmethod
    def get(self, resource: str=None, split: str=None, transform=None, **kwargs):
        raise NotImplementedError
    

from torch_geometric.transforms import BaseTransform, Compose
from src.data.transforms.core import TransformAdapter


def transforms_to_pipeline(transforms, **kwargs):
    if transforms is None:
        return None

    if not isinstance(transforms, list):
        transforms = [transforms]

    pipeline = []

    for t in transforms:
        if isinstance(t, BaseTransform):
            pipeline.append(t)
        elif isinstance(t, TransformAdapter):
            pipeline.append(t.instantiate(**kwargs))
        else:
            raise ValueError(f'Invalid transform: {t}')
        
    return Compose(pipeline)