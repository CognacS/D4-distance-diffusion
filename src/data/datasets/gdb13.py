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
from src.data.datasets.molecular import MolecularGraphsDataset, SmilesDataset, MolecularDataset

from copy import copy

# all dataset content
GDB13_DATA_URL = 'https://zenodo.org/record/5172018/files/gdb13.rand1M.smi.gz?download=1&ref=gdb.unibe.ch'


# content of zip file
GDB13_RAW_SMI = 'gdb13.rand1M.smi'

DEFAULT_DATASET_PATH_GDB13 = osp.join(DEFAULT_DATASET_PATH, 'gdb13')


class RawProcWorker:
    
    def __init__(self, kekulize: bool = True, pre_filter: Optional[Callable] = None):
        self.kekulize = kekulize
        self.pre_filter = pre_filter
    
    
    def __call__(self, mol: Chem.Mol) -> tuple:
        
        try:

            if mol is None:     # skip if molecule is None (e.g., sanitization failed)
                skipped_sanitization += 1
                return (None, 'sanitization failed')

            if self.kekulize:
                mol = molutils.kekulize_molecule(mol)
                
            if self.pre_filter is not None and not self.pre_filter(mol):
                skipped_prefilter += 1
                return (None, 'pre-filter failed')

            return (mol, None)
        
        except Exception as e:
            # print(f'Error processing molecule: {e}')
            return (None, str(e))


class GDB13Raw(RawDataset):
    """ GDB13 dataset class for raw data.
    """

    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            sanitize: bool = True,
            remove_hydrogens: bool = True,
            kekulize: bool = True,
            subset_size: Optional[int] = None,
            num_workers: int = 1,
            pre_transform=None,
            pre_filter=None
        ):
        
        self.sanitize = sanitize
        self.remove_hydrogens = remove_hydrogens
        self.kekulize = kekulize
        self.subset_size = subset_size
        self.num_workers = num_workers

        if root is None:
            root = DEFAULT_DATASET_PATH_GDB13

        super().__init__(root, split=split, pre_transform=pre_transform, pre_filter=pre_filter)

        if not hasattr(self, 'mols'):
            self.mols = self.load(self.raw_paths[0])
            self.stats = self.load(self.raw_paths[1])
            self.atom_types = self.stats['atom_types']
            self.bond_types = self.stats['bond_types']
            self.charges = self.stats['charges'] if 'charges' in self.stats else None


    def subset_from(self, indices: List[int], name: str) -> Dataset:
        """Create a subset of the dataset with the indices provided, at the folder
        name provided. This is useful for splitting the dataset into train, test, and validation
        """

        subset = copy(self)
        subset.root = self.root
        subset.split = name
        makedirs(subset.raw_dir)
        subset.mols = [self.mols[i] for i in indices]

        # save data
        subset.save(subset.mols, subset.raw_paths[0])

        # get statistics
        stats_new = molutils.get_molecule_stats(subset.mols)
        stats_new['atom_types'] = self.atom_types # use old atom types
        stats_new['bond_types'] = self.bond_types # use old bond types
        stats_new['charges'] = self.charges # use old charges
        subset.stats = stats_new
        subset.atom_types = self.atom_types
        subset.bond_types = self.bond_types
        subset.charges = self.charges
        
        # store data in files
        subset.save(subset.stats, subset.raw_paths[1])

        return subset

    
    @property
    def raw_file_names(self):
        return ['mols.pkl', 'stats.json']
    
    
    def download(self):
        # download gz file
        gz_file = download_url(GDB13_DATA_URL, self.raw_dir)
        
        # extract data in gz to folder
        extract_gz(gz_file, self.raw_dir)
        smiles_file = osp.join(self.raw_dir, GDB13_RAW_SMI)

        # read smiles
        self.mols = molutils.read_molecules(smiles_file, self.sanitize, self.remove_hydrogens)
        
        # last round of processing
        self.mols = self._prepare_data(self.mols)

        # filter data if needed
        # if self.pre_filter is not None:
        #     len_before = len(self.mols)
        #     self.mols = [d for d in self.mols if self.pre_filter(d)]
        #     len_after = len(self.mols)
        #     if len_after < len_before:
        #         print(f'Filtered {len_before - len_after} molecules from GDB13 dataset using pre_filter')
            
        if self.subset_size is not None:
            # shuffle mols
            import random
            random.shuffle(self.mols)
            # take only the first subset_size molecules
            self.mols = self.mols[:self.subset_size]

        # apply pre_transform if needed
        if self.pre_transform is not None:
            self.mols = [self.pre_transform(d) for d in self.mols]

        # save data
        self.save(self.mols, self.raw_paths[0])

        # get statistics
        self.stats = molutils.get_molecule_stats(self.mols)
        self.atom_types = self.stats['atom_types']
        self.bond_types = self.stats['bond_types']
        self.charges = self.stats['charges']
        
        # store data in files
        self.save(self.stats, self.raw_paths[1])
        

        
    def _prepare_data(self, mols):
        
        mols = [mol for mol in mols]
        
        chunksize = (len(mols) // self.num_workers // 20) if self.num_workers > 0 else len(mols)
        
        results = process_map(RawProcWorker(kekulize=self.kekulize, pre_filter=self.pre_filter),
                mols, max_workers=self.num_workers, desc='Processing molecules', chunksize=chunksize)
        
        skipped_sanitization = 0
        skipped_prefilter = 0
        errors = []
        def select_valid_mols(result):
            mol, error = result
            if mol is None:
                if error == 'sanitization failed':
                    skipped_sanitization += 1
                elif error == 'pre-filter failed':
                    skipped_prefilter += 1
                else:
                    errors.append(error)
                return False
            return True
        
        final_mols = [mol for mol, error in results if select_valid_mols((mol, error))]
        
        print(f'Sanitization skipped {skipped_sanitization} molecules')
        print(f'Pre-filter skipped {skipped_prefilter} molecules')
        print(f'Errors:', errors)
            
        return final_mols


    def __len__(self):
        return len(self.mols)
        
    def __getitem__(self, idx):
        return self.mols[idx]
    

class GDB13(MolecularGraphsDataset):

    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            sanitize: bool = True,
            remove_hydrogens: bool = True,
            kekulize: bool = True,
            hard_remove_hydrogens: bool = True,
            subset_size: Optional[int] = None,
            include_pos: bool = False,
            include_charges: bool = False,
            num_workers: int = 1,
            pre_transform_raw=None,
            pre_filter_raw=None,
            transform=None,
            pre_transform=None,
            pre_filter=None
        ):

        if root is None:
            root = DEFAULT_DATASET_PATH_GDB13

        # create raw dataset
        raw_dataset = GDB13Raw(
            root, sanitize=sanitize,
            remove_hydrogens=remove_hydrogens, kekulize=kekulize, subset_size=subset_size,
            pre_transform=pre_transform_raw, pre_filter=pre_filter_raw, num_workers=num_workers
        )

        super().__init__(
            root, split=split, raw_mol_dataset=raw_dataset,
            atom_types=raw_dataset.atom_types, bond_types=raw_dataset.bond_types,
            charges=raw_dataset.charges,
            hard_remove_hydrogens=hard_remove_hydrogens,
            include_pos=include_pos, include_charges=include_charges,
            transform=transform, pre_transform=pre_transform, pre_filter=pre_filter,
            num_workers=num_workers
        )

class GDB13Smiles(SmilesDataset):
    
    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            sanitize: bool = True,
            remove_hydrogens: bool = True,
            kekulize: bool = True,
            subset_size: Optional[int] = None,
            pre_transform_raw=None,
            pre_filter_raw=None,
            pre_transform=None,
            pre_filter=None
        ):

        if root is None:
            root = DEFAULT_DATASET_PATH_GDB13

        # create raw dataset
        raw_dataset = GDB13Raw(
            root, sanitize=sanitize,
            remove_hydrogens=remove_hydrogens, kekulize=kekulize, subset_size=subset_size,
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
            sanitize: bool = True,
            remove_hydrogens: bool = True,
            kekulize: bool = True,
            hard_remove_hydrogens: bool = True,
            subset_size: Optional[int] = None,
            include_pos: bool = False,
            include_charges: bool = False,
            num_workers: int = 1,
            pre_transform=None,
            pre_filter=None,
            pre_transform_raw=None,
            pre_filter_raw=None
        ):

        super().__init__()
        
        self.root = root
        
        self.gdb13_cfg = {
            'sanitize': sanitize,
            'remove_hydrogens': remove_hydrogens,
            'kekulize': kekulize,
            'hard_remove_hydrogens': hard_remove_hydrogens,
            'subset_size': subset_size,
            'include_pos': include_pos,
            'include_charges': include_charges,
            'num_workers': num_workers,  # default to 1 worker, can be changed later
            'pre_transform_raw': pre_transform_raw,
            'pre_filter_raw': pre_filter_raw
        }
        self.smiles_cfg = {
            'sanitize': sanitize,
            'remove_hydrogens': remove_hydrogens,
            'kekulize': kekulize,
            'subset_size': subset_size,
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