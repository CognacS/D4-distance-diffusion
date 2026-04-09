from typing import List, Optional, Dict, Callable
import os.path as osp
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from collections import OrderedDict

from torch.utils.data import Dataset

from rdkit import Chem

import src.data.utils.csv as csvutils
import src.data.utils.molecular as molutils
from torch_geometric.data import extract_zip, download_url, extract_gz
#from torch_geometric.data.dataset import makedirs
from torch_geometric.io.fs import makedirs
from torch_geometric.datasets.qm9 import conversion

from src.data.datasets.core import RawDataset, DataResources, DatasetException, DEFAULT_DATASET_PATH, DEFAULT_SPLITS
from src.data.datasets.molecular import MolecularGraphsDataset, SmilesDataset, ExtendedMolecularDatasetRaw
from src.data.datasets.atom_types_representation import AUXILIARY_NODE_STATES_STATS_KEY

from copy import copy

# all dataset content
GDB13_DATA_URL = 'https://zenodo.org/record/5172018/files/gdb13.rand1M.smi.gz?download=1&ref=gdb.unibe.ch'


# content of zip file
GDB13_RAW_SMI = 'gdb13.rand1M.smi'

DEFAULT_DATASET_PATH_GDB13 = osp.join(DEFAULT_DATASET_PATH, 'gdb13')


class GDB13Raw(ExtendedMolecularDatasetRaw):
    """ GDB13 dataset class for raw data.
    """

    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            sanitize: bool = False,
            remove_hydrogens: bool = False,
            kekulize: bool = False,
            compute_3d_conformer: bool = False,
            properties_computer_function: Optional[Callable] = None,
            num_workers: int = 0,
            chunksize: Optional[int] = None,
            pre_transform=None,
            pre_filter=None,
            atom_types_repr: str = 'default',
        ):

        if root is None:
            root = DEFAULT_DATASET_PATH_GDB13

        super().__init__(
            root=root, split=split, sanitize=sanitize,
            remove_hydrogens=remove_hydrogens, kekulize=kekulize,
            compute_3d_conformer=compute_3d_conformer,
            properties_computer_function=properties_computer_function,
            num_workers=num_workers, chunksize=chunksize,
            pre_transform=pre_transform, pre_filter=pre_filter,
            atom_types_repr=atom_types_repr,
        )
        
        self.load_data(
            mols_path=self.raw_paths[0],
            props_path=self.raw_paths[1],
            stats_path=self.raw_paths[2]
        )
        
    def subset_from(self, indices: List[int], name: str):
        """Create a subset of the dataset with the indices provided, at the folder
        name provided. This is useful for splitting the dataset into train, test, and validation
        """
        
        return super().subset_from(
            indices, name,
            mols_path=self.raw_paths[0],
            props_path=self.raw_paths[1],
            stats_path=self.raw_paths[2]
        )

    
    @property
    def raw_file_names(self):
        return ['mols.pkl', 'props.json', 'stats.json']


    def download(self):
        # download gz file
        gz_file = download_url(GDB13_DATA_URL, self.raw_dir)
        
        # extract data in gz to folder
        extract_gz(gz_file, self.raw_dir)
        smiles_file = osp.join(self.raw_dir, GDB13_RAW_SMI)

        # read smiles
        smiles = molutils.read_molecules(smiles_file, self.sanitize, self.remove_hydrogens)
        smiles = [s for s in tqdm(smiles, desc='Reading SMILES') if s is not None]
        props = [OrderedDict()] * len(smiles)  # no starting properties available
        
        self.mols, self.props = self.preprocess_molecules(
            smiles, props, mols_path=self.raw_paths[0], props_path=self.raw_paths[1]
        )

        # get statistics
        self.stats = molutils.get_molecule_stats(self.mols, self.atom_types_repr)
        self.atom_types = self.stats['atom_types']
        self.bond_types = self.stats['bond_types']
        self.charges = self.stats['charges']
        self.auxiliary_node_state_values = self.stats.get(AUXILIARY_NODE_STATES_STATS_KEY)
        
        # store data in files
        self.save(self.stats, self.raw_paths[2])


class GDB13(MolecularGraphsDataset):

    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            sanitize: bool = False,
            remove_hydrogens: bool = True,
            kekulize: bool = True,
            hard_remove_hydrogens: bool = False,
            include_pos: bool = False,
            include_charges: bool = False,
            num_workers: int = 0,
            chunksize: Optional[int] = None,
            properties_computer_function: Optional[Callable] = None,
            pre_transform_raw=None,
            pre_filter_raw=None,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            atom_types_repr: str = 'default',
        ):

        if root is None:
            root = DEFAULT_DATASET_PATH_GDB13

        # create raw dataset
        raw_dataset = GDB13Raw(
            root=root, sanitize=sanitize,
            remove_hydrogens=remove_hydrogens, kekulize=kekulize,
            compute_3d_conformer=include_pos,
            properties_computer_function=properties_computer_function,
            num_workers=num_workers, chunksize=chunksize,
            pre_transform=pre_transform_raw, pre_filter=pre_filter_raw,
            atom_types_repr=atom_types_repr
        )

        super().__init__(
            root, split=split, raw_mol_dataset=raw_dataset,
            atom_types=raw_dataset.atom_types, bond_types=raw_dataset.bond_types,
            charges=raw_dataset.charges,
            hard_remove_hydrogens=hard_remove_hydrogens,
            include_pos=include_pos, include_charges=include_charges,
            transform=transform, pre_transform=pre_transform, pre_filter=pre_filter,
            atom_types_repr=atom_types_repr
            #num_workers=num_workers, chunksize=chunksize
        )

class GDB13Smiles(SmilesDataset):
    
    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            sanitize: bool = True,
            remove_hydrogens: bool = True,
            kekulize: bool = True,
            include_pos: bool = False,
            properties_computer_function: Optional[Callable] = None,
            num_workers: int = 0,
            chunksize: Optional[int] = None,
            pre_transform_raw=None,
            pre_filter_raw=None,
            pre_transform=None,
            pre_filter=None
        ):

        if root is None:
            root = DEFAULT_DATASET_PATH_GDB13

        # create raw dataset
        raw_dataset = GDB13Raw(
            root=root, sanitize=sanitize,
            remove_hydrogens=remove_hydrogens, kekulize=kekulize,
            compute_3d_conformer=include_pos,
            properties_computer_function=properties_computer_function,
            num_workers=num_workers, chunksize=chunksize,
            pre_transform=pre_transform_raw, pre_filter=pre_filter_raw
        )

        super().__init__(
            root, split=split, raw_mol_dataset=raw_dataset,
            pre_transform=pre_transform, pre_filter=pre_filter
        )


from src.data.datasets.split import random_split_dataset
from src.data.datasets import reg_dataresources

@reg_dataresources.register('gdb13')
class GDB13Resources(DataResources):

    def __init__(
            self,
            random_splits: Dict,
            root: Optional[str] = None,
            sanitize: bool = False,
            remove_hydrogens: bool = True,
            kekulize: bool = True,
            hard_remove_hydrogens: bool = False,
            include_pos: bool = False,
            include_charges: bool = False,
            num_workers: int = 0,
            chunksize: Optional[int] = None,
            pre_transform=None,
            pre_filter=None,
            pre_transform_raw=None,
            pre_filter_raw=None,
            atom_types_repr: str = 'default',
        ):

        super().__init__()
        
        self.root = root
        
        self.gdb13_cfg = {
            'sanitize': sanitize,
            'remove_hydrogens': remove_hydrogens,
            'kekulize': kekulize,
            'hard_remove_hydrogens': hard_remove_hydrogens,
            'include_pos': include_pos,
            'include_charges': include_charges,
            'num_workers': num_workers,
            'chunksize': chunksize,
            'pre_transform_raw': pre_transform_raw,
            'pre_filter_raw': pre_filter_raw,
            'atom_types_repr': atom_types_repr
        }
        self.smiles_cfg = {
            'sanitize': sanitize,
            'remove_hydrogens': remove_hydrogens,
            'kekulize': kekulize,
            'pre_transform_raw': pre_transform_raw,
            'pre_filter_raw': pre_filter_raw
        }

        if random_splits is None:
            random_splits = DEFAULT_SPLITS
        self.random_splits = random_splits

        self.preproc = {
            'pre_transform': pre_transform,
            'pre_filter': pre_filter
        }

        self._prepared = False


    def prepare_data(self):

        ds = GDB13(self.root, **self.gdb13_cfg)
        ds_smiles = GDB13Smiles(self.root, **self.smiles_cfg)

        self.decoder = ds.mol_to_torch_converter
        self.info_total = ds.stats

        # if there is any pre_transform, resolve any transform adapter
        self.preproc['pre_transform'] = self.transforms_to_pipeline(self.preproc['pre_transform'])
        self.preproc['pre_filter'] = self.filters_to_pipeline(self.preproc['pre_filter'])

        try: # try to get the split datasets

            dss = {split: [
                    GDB13(self.root, split=split, **self.preproc, **self.gdb13_cfg),
                    GDB13Smiles(self.root, split=split, **self.smiles_cfg)
                ] for split in self.random_splits
            }

        except DatasetException: # if not possible, create the splits
            print('Creating random splits for GDB13 graphs and SMILES')

            # reload dataset with preprocessing
            if self.preproc['pre_transform'] is not None:
                print('Applying preprocessing to the whole dataset')
                ds.reapply_pre_transform(self.preproc['pre_transform'], self.preproc['pre_filter'])

            dss = random_split_dataset([ds, ds_smiles], self.random_splits)

        self.info = {split: self.wrap_dataset(d[0]).stats for split, d in dss.items()}

        self._prepared = True


    def get(self, resource: str=None, split: str=None, transform=None):
        if not self._prepared:
            self.prepare_data()

        if resource == 'dataset':
            return self.wrap_dataset(GDB13(self.root, split=split, **self.preproc, **self.gdb13_cfg), transform=transform)
        elif resource == 'smiles':
            return GDB13Smiles(self.root, split=split, **self.smiles_cfg)
        elif resource == 'decoder':
            return self.decoder
        elif resource == 'info':
            return self.info[split] if split in self.info else self.info_total
        else:
            raise ValueError(f'Resource {resource} not found for GDB13 dataset, choose between "dataset" and "smiles"')
        

    def __repr__(self):
        return f'{self.__class__.__name__}[resources=[dataset, smiles, decoder, info], splits={list(self.random_splits.keys())}]'
    

@reg_dataresources.register('gdb13_mmff')
class GDB13Resources(DataResources):

    def __init__(
            self,
            random_splits: Dict,
            root: Optional[str] = None,
            sanitize: bool = False,
            remove_hydrogens: bool = True,
            kekulize: bool = True,
            hard_remove_hydrogens: bool = False,
            include_pos: bool = False,
            include_charges: bool = False,
            num_workers: int = 0,
            chunksize: Optional[int] = None,
            pre_transform=None,
            pre_filter=None,
            pre_transform_raw=None,
            pre_filter_raw=None,
            atom_types_repr: str = 'default',
        ):

        super().__init__()
        
        self.root = root
        
        self.gdb13_cfg = {
            'sanitize': sanitize,
            'remove_hydrogens': remove_hydrogens,
            'kekulize': kekulize,
            'hard_remove_hydrogens': hard_remove_hydrogens,
            'include_pos': include_pos,
            'include_charges': include_charges,
            'num_workers': num_workers,
            'chunksize': chunksize,
            'pre_transform_raw': pre_transform_raw,
            'pre_filter_raw': pre_filter_raw,
            'atom_types_repr': atom_types_repr
        }
        self.smiles_cfg = {
            'sanitize': sanitize,
            'remove_hydrogens': remove_hydrogens,
            'kekulize': kekulize,
            'pre_transform_raw': pre_transform_raw,
            'pre_filter_raw': pre_filter_raw
        }

        if random_splits is None:
            random_splits = DEFAULT_SPLITS
        self.random_splits = random_splits

        self.preproc = {
            'pre_transform': pre_transform,
            'pre_filter': pre_filter
        }

        self._prepared = False


    def prepare_data(self):

        ds = GDB13(self.root, **self.gdb13_cfg)
        ds_smiles = GDB13Smiles(self.root, **self.smiles_cfg)

        self.decoder = ds.mol_to_torch_converter
        self.info_total = ds.stats

        # if there is any pre_transform, resolve any transform adapter
        self.preproc['pre_transform'] = self.transforms_to_pipeline(self.preproc['pre_transform'])
        self.preproc['pre_filter'] = self.filters_to_pipeline(self.preproc['pre_filter'])

        try: # try to get the split datasets

            dss = {split: [
                    GDB13(self.root, split=split, **self.preproc, **self.gdb13_cfg),
                    GDB13Smiles(self.root, split=split, **self.smiles_cfg)
                ] for split in self.random_splits
            }

        except DatasetException: # if not possible, create the splits
            print('Creating random splits for GDB13 graphs and SMILES')

            # reload dataset with preprocessing
            if self.preproc['pre_transform'] is not None:
                print('Applying preprocessing to the whole dataset')
                ds.reapply_pre_transform(self.preproc['pre_transform'], self.preproc['pre_filter'])

            dss = random_split_dataset([ds, ds_smiles], self.random_splits)

        self.info = {split: self.wrap_dataset(d[0]).stats for split, d in dss.items()}

        self._prepared = True


    def get(self, resource: str=None, split: str=None, transform=None):
        if not self._prepared:
            self.prepare_data()

        if resource == 'dataset':
            return self.wrap_dataset(GDB13(self.root, split=split, **self.preproc, **self.gdb13_cfg), transform=transform)
        elif resource == 'smiles':
            return GDB13Smiles(self.root, split=split, **self.smiles_cfg)
        elif resource == 'decoder':
            return self.decoder
        elif resource == 'info':
            return self.info[split] if split in self.info else self.info_total
        else:
            raise ValueError(f'Resource {resource} not found for GDB13 dataset, choose between "dataset" and "smiles"')
        

    def __repr__(self):
        return f'{self.__class__.__name__}[resources=[dataset, smiles, decoder, info], splits={list(self.random_splits.keys())}]'
    

@reg_dataresources.register('gdb13_atom_details')
class GDB13Resources(DataResources):

    def __init__(
            self,
            random_splits: Dict,
            root: Optional[str] = None,
            sanitize: bool = False,
            remove_hydrogens: bool = True,
            kekulize: bool = True,
            hard_remove_hydrogens: bool = False,
            include_pos: bool = False,
            include_charges: bool = False,
            num_workers: int = 0,
            chunksize: Optional[int] = None,
            pre_transform=None,
            pre_filter=None,
            pre_transform_raw=None,
            pre_filter_raw=None,
            atom_types_repr: str = 'default',
        ):

        super().__init__()
        
        self.root = root
        
        self.gdb13_cfg = {
            'sanitize': sanitize,
            'remove_hydrogens': remove_hydrogens,
            'kekulize': kekulize,
            'hard_remove_hydrogens': hard_remove_hydrogens,
            'include_pos': include_pos,
            'include_charges': include_charges,
            'num_workers': num_workers,
            'chunksize': chunksize,
            'pre_transform_raw': pre_transform_raw,
            'pre_filter_raw': pre_filter_raw,
            'atom_types_repr': atom_types_repr
        }
        self.smiles_cfg = {
            'sanitize': sanitize,
            'remove_hydrogens': remove_hydrogens,
            'kekulize': kekulize,
            'pre_transform_raw': pre_transform_raw,
            'pre_filter_raw': pre_filter_raw
        }

        if random_splits is None:
            random_splits = DEFAULT_SPLITS
        self.random_splits = random_splits

        self.preproc = {
            'pre_transform': pre_transform,
            'pre_filter': pre_filter
        }

        self._prepared = False


    def prepare_data(self):

        ds = GDB13(self.root, **self.gdb13_cfg)
        ds_smiles = GDB13Smiles(self.root, **self.smiles_cfg)

        self.decoder = ds.mol_to_torch_converter
        self.info_total = ds.stats

        # if there is any pre_transform, resolve any transform adapter
        self.preproc['pre_transform'] = self.transforms_to_pipeline(self.preproc['pre_transform'])
        self.preproc['pre_filter'] = self.filters_to_pipeline(self.preproc['pre_filter'])

        try: # try to get the split datasets

            dss = {split: [
                    GDB13(self.root, split=split, **self.preproc, **self.gdb13_cfg),
                    GDB13Smiles(self.root, split=split, **self.smiles_cfg)
                ] for split in self.random_splits
            }

        except DatasetException: # if not possible, create the splits
            print('Creating random splits for GDB13 graphs and SMILES')

            # reload dataset with preprocessing
            if self.preproc['pre_transform'] is not None:
                print('Applying preprocessing to the whole dataset')
                ds.reapply_pre_transform(self.preproc['pre_transform'], self.preproc['pre_filter'])

            dss = random_split_dataset([ds, ds_smiles], self.random_splits)

        self.info = {split: self.wrap_dataset(d[0]).stats for split, d in dss.items()}

        self._prepared = True


    def get(self, resource: str=None, split: str=None, transform=None):
        if not self._prepared:
            self.prepare_data()

        if resource == 'dataset':
            return self.wrap_dataset(GDB13(self.root, split=split, **self.preproc, **self.gdb13_cfg), transform=transform)
        elif resource == 'smiles':
            return GDB13Smiles(self.root, split=split, **self.smiles_cfg)
        elif resource == 'decoder':
            return self.decoder
        elif resource == 'info':
            return self.info[split] if split in self.info else self.info_total
        else:
            raise ValueError(f'Resource {resource} not found for GDB13 dataset, choose between "dataset" and "smiles"')
        

    def __repr__(self):
        return f'{self.__class__.__name__}[resources=[dataset, smiles, decoder, info], splits={list(self.random_splits.keys())}]'