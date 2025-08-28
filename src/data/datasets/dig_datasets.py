from typing import List, Optional, Tuple, Dict, Any, Type, Union
from tqdm import tqdm
from collections import OrderedDict

import os
import os.path as osp

from src.data.datasets.core import RawDataset, DatasetException, DataResources, DEFAULT_DATASET_PATH
from src.data.datasets.split import split_dataset, random_split_dataset

from torch_geometric.data import download_url
from torch_geometric.io.fs import makedirs

import src.data.utils.csv as csvutils

from copy import copy

DIG_RAW_REPO_URL = 'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/'
SUPPORTED_DATASETS = [
    'qm9',
    'zinc250k',
]

class BaseDigSmilesRaw(RawDataset):

    def __init__(
            self,
            which_dataset: str,
            root: Optional[str] = None,
            split: Optional[str] = None,
            return_only_smiles: bool = False,
            pre_transform=None,
            pre_filter=None
        ):

        if which_dataset not in SUPPORTED_DATASETS:
            raise ValueError(f'Dataset {which_dataset} not supported. Supported datasets are: {SUPPORTED_DATASETS}')
        self.which_dataset = which_dataset

        if root is None:
            root = osp.join(DEFAULT_DATASET_PATH, which_dataset + '-dig')

        self.return_only_smiles = return_only_smiles

        super().__init__(root, split=split, pre_transform=pre_transform, pre_filter=pre_filter)

        if not hasattr(self, 'smiles'):
            self.smiles = self.load(self.raw_paths[0])
            self.props = self.load(self.raw_paths[1])


    def subset_from(self, indices: List[int], name: str):

        subset = copy(self)
        subset.root = self.root
        subset.split = name

        makedirs(subset.raw_dir)

        subset.smiles = [self.smiles[i] for i in indices]
        subset.props = [self.props[i] for i in indices] if len(self.props) > 0 else []

        subset.save(subset.smiles, subset.raw_paths[0])
        subset.save(subset.props, subset.raw_paths[1])
        subset.save(self.load(self.raw_paths[2]), subset.raw_paths[2]) # save test indices

        return subset

    
    @property
    def raw_file_names(self):
        return ['smiles.json', 'props.json', 'test_idx.json']
    
    @property
    def other_file_names(self):
        l = super().other_file_names
        return l + [self.which_dataset + '.csv']
    

    def process_csv(self, header: List[str], ids: List[int], rows: List[OrderedDict]) -> Tuple[List[str], List[float]]|List[str]:
        # this method should return either:
        # - a tuple with a list of smiles and a list of properties
        # - a list of smiles (if no properties are available)
        raise NotImplementedError('This method should be implemented in the subclass')
    
    def process_test_indices (self, struct: Any) -> List[int]:
        # this method should return a list of ints from the provided structure
        raise NotImplementedError('This method should be implemented in the subclass')


    def get_test_indices(self) -> List[int]:
        struct = self.load(self.raw_paths[2])
        return self.process_test_indices(struct)


    def download(self):
        # download smiles file
        smiles_url = DIG_RAW_REPO_URL + self.which_dataset + '.csv'
        smiles_file = download_url(smiles_url, self.raw_dir)

        # download test split indices
        test_idx_url = DIG_RAW_REPO_URL + 'valid_idx_' + self.which_dataset + '.json'
        test_idx_file = download_url(test_idx_url, self.raw_dir)

        # rename index file to make it general
        test_idx_file_new = osp.join(self.raw_dir, 'test_idx.json')
        if not osp.exists(test_idx_file_new):
            os.rename(test_idx_file, test_idx_file_new)

        # read smiles csv file
        header, ids, rows = csvutils.read_csv_with_header_and_index(smiles_file)

        # process csv file - should be implemented in the subclass
        ret = self.process_csv(header, ids, rows)

        # if properties are available, unpack them
        if isinstance(ret, tuple):
            self.smiles, self.props = ret
        else:
            self.smiles, self.props = ret, []

        self.save(self.smiles, self.raw_paths[0])
        self.save(self.props, self.raw_paths[1])


    def __len__(self):
        return len(self.smiles)
        
    def __getitem__(self, idx):
        if self.return_only_smiles:
            return self.smiles[idx]
        else:
            s = self.smiles[idx]
            p = self.props[idx] if len(self.props) > 0 else OrderedDict()
            return s, p
        


class BaseDigResources(DataResources):

    def __init__(
            self,
            random_splits: Dict,
            dataset_cfg: Dict,
            smiles_cfg: Dict,
            dataset_cls,
            smiles_cls,
            root: Optional[str] = None,
            pre_transform=None,
            pre_filter=None,
        ):
        super().__init__()
        
        self.root = root
        
        self.dataset_cfg = dataset_cfg
        self.smiles_cfg = smiles_cfg
        self.random_splits = random_splits

        self.dataset_cls = dataset_cls
        self.smiles_cls = smiles_cls

        if 'test' in self.random_splits:
            print('Warning: test split will be created from a predefined index. The provided test split will be ignored.')
            self.random_splits.pop('test')

        self.preproc = {
            'pre_transform': pre_transform,
            'pre_filter': pre_filter
        }

        self._prepared = False


    def prepare_data(self):

        ds = self.dataset_cls(self.root, **self.dataset_cfg)
        ds_smiles = self.smiles_cls(self.root, **self.smiles_cfg)

        self.decoder = ds.mol_to_torch_converter
        self.info_total = ds.stats

        # if there is any pre_transform, solve any transform adapter
        self.preproc['pre_transform'] = self.transforms_to_pipeline(self.preproc['pre_transform'])

        try: # try to get the split datasets

            dss = {split: [
                    self.dataset_cls(self.root, split=split, **self.preproc, **self.dataset_cfg),
                    self.smiles_cls(self.root, split=split, **self.smiles_cfg)
                ] for split in self.random_splits
            }

            dss['test'] = [
                self.dataset_cls(self.root, split='test', **self.preproc, **self.dataset_cfg),
                self.smiles_cls(self.root, split='test', **self.smiles_cfg)
            ]

        except DatasetException: # if not possible, create the splits

            print('Creating random splits for graphs and SMILES')

            # reload dataset with preprocessing
            if self.preproc['pre_transform'] is not None:
                ds.delete()
                ds = self.dataset_cls(self.root, **self.preproc, **self.dataset_cfg)

            test_indices = ds_smiles.get_test_indices()
            other_indices = [i for i in range(len(ds)) if i not in test_indices]

            dss_train_test = split_dataset(
                [ds, ds_smiles],
                {'others': other_indices, 'test': test_indices}
            )

            dss = random_split_dataset(dss_train_test['others'], self.random_splits)
            # delete others
            [curr_ds.delete() for curr_ds in dss_train_test['others']]
            dss['test'] = dss_train_test['test']
        

        self.info = {split: self.wrap_dataset(d[0]).stats for split, d in dss.items()}

        self._prepared = True


    def get_dataset(self, split: str, transform=None):
        return self.wrap_dataset(
            self.dataset_cls(self.root, split=split, **self.preproc, **self.dataset_cfg),
            transform=transform
        )
    
    def get_smiles(self, split: str):
        return self.smiles_cls(self.root, split=split, **self.smiles_cfg)


    def get(self, resource: str=None, split: str=None, transform=None):
        if not self._prepared:
            self.prepare_data()

        if resource == 'dataset':
            return self.get_dataset(split, transform=transform)
        elif resource == 'smiles':
            return self.get_smiles(split)
        elif resource == 'decoder':
            return self.decoder
        elif resource == 'info':
            return self.info[split] if split in self.info else self.info_total
        else:
            raise ValueError(f'Resource {resource} not found for {self.dataset_cls.__name__}, choose between "dataset" and "smiles"')
        

    def __repr__(self):
        return f'{self.__class__.__name__}[resources=[dataset, smiles, decoder, info], splits={list(self.random_splits.keys())}]'