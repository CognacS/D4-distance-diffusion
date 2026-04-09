from typing import List, Optional, Dict
import os.path as osp
from tqdm import tqdm
from collections import OrderedDict

from torch.utils.data import Dataset

import src.data.utils.csv as csvutils
import src.data.utils.molecular as molutils
from torch_geometric.data import extract_zip, download_url
#from torch_geometric.data.dataset import makedirs
from torch_geometric.io.fs import makedirs
from torch_geometric.datasets.qm9 import conversion

from src.data.datasets.core import RawDataset, DataResources, DatasetException, DEFAULT_DATASET_PATH, DEFAULT_SPLITS
from src.data.datasets.molecular import MolecularGraphsDataset, SmilesDataset
from src.data.datasets.atom_types_representation import AUXILIARY_NODE_STATES_STATS_KEY

from copy import copy

# all dataset content
QM9_ZIP_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip'
# the readme is just for reference
QM9_README_URL = 'https://ndownloader.figshare.com/files/3195404'

# content of zip file
QM9_RAW_SDF = 'gdb9.sdf'
QM9_RAW_CSV = 'gdb9.sdf.csv'
QM9_SKIP_FILE = '3195404'

DEFAULT_DATASET_PATH_QM9 = osp.join(DEFAULT_DATASET_PATH, 'qm9')
    

def apply_unit_conversion(header: List[str], properties: List[OrderedDict]) -> List[OrderedDict]:

    # get new order of properties
    new_header = header[3:] + header[:3]

    # process properties with conversion factors from torch_geometric.datasets.qm9
    properties = [OrderedDict([(k, prop[k] * c.item()) for k, c in zip(new_header, conversion)]) for prop in properties]

    return properties
    


def prepare_skip_list(filepath: str) -> List[int]:
    with open(filepath, 'r') as f:
        skip_list = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

    return skip_list


class QM9Raw(RawDataset):
    """ QM9 dataset class for raw data.
    """

    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            sanitize: bool = False,
            remove_hydrogens: bool = True,
            kekulize: bool = True,
            pre_transform=None,
            pre_filter=None,
            atom_types_repr: str= 'default'
        ):

        self.sanitize = sanitize
        self.remove_hydrogens = remove_hydrogens
        self.kekulize = kekulize
        self.atom_types_repr = atom_types_repr

        if root is None:
            root = DEFAULT_DATASET_PATH_QM9

        super().__init__(root, split=split, pre_transform=pre_transform, pre_filter=pre_filter)

        if not hasattr(self, 'mols'):
            self.mols = self.load(self.raw_paths[0])
            self.props = self.load(self.raw_paths[1])
            self.stats = self.load(self.raw_paths[2])
            self.atom_types = self.stats['atom_types']
            self.bond_types = self.stats['bond_types']
            self.charges = self.stats['charges'] if 'charges' in self.stats else None
            self.auxiliary_node_state_values = self.stats.get(AUXILIARY_NODE_STATES_STATS_KEY)


    
    def subset_from(self, indices: List[int], name: str) -> Dataset:
        """Create a subset of the dataset with the indices provided, at the folder
        name provided. This is useful for splitting the dataset into train, test, and validation
        """

        subset = copy(self)
        subset.root = self.root
        subset.split = name
        makedirs(subset.raw_dir)
        subset.mols = [self.mols[i] for i in indices]
        subset.props = [self.props[i] for i in indices]

        # save data
        subset.save(subset.mols, subset.raw_paths[0])
        subset.save(subset.props, subset.raw_paths[1])

        # get statistics
        stats_new = molutils.get_molecule_stats(subset.mols, self.atom_types_repr)
        stats_new['atom_types'] = self.atom_types # use old atom types
        stats_new['bond_types'] = self.bond_types # use old bond types
        stats_new['charges'] = self.charges # use old charges
        if self.auxiliary_node_state_values is not None:
            stats_new[AUXILIARY_NODE_STATES_STATS_KEY] = self.auxiliary_node_state_values
        subset.stats = stats_new
        subset.atom_types = self.atom_types
        subset.bond_types = self.bond_types
        subset.charges = self.charges
        subset.auxiliary_node_state_values = self.auxiliary_node_state_values
        
        # store data in files
        subset.save(subset.stats, subset.raw_paths[2])

        return subset


    
    @property
    def raw_file_names(self):
        return [
            'mols.pkl', 'props.json', 'stats.json']
    
    @property
    def other_file_names(self):
        l = super().other_file_names
        return l + ['qm9.zip', 'gdb9.sdf.csv', '3195404', 'gdb9.sdf', 'QM9_README']


    def download(self):
        # download data from URLs
        zip_file = download_url(QM9_ZIP_URL, self.raw_dir)
        readme_file = download_url(QM9_README_URL, self.raw_dir)

        # extract data in zip to folder
        extract_zip(zip_file, self.raw_dir)
        molecules_file = osp.join(self.raw_dir, QM9_RAW_SDF)
        properties_file = osp.join(self.raw_dir, QM9_RAW_CSV)
        skip_file = osp.join(self.raw_dir, QM9_SKIP_FILE)

        # read molecules
        mols = molutils.read_molecules(molecules_file, self.sanitize, self.remove_hydrogens)

        # read molecular properties
        header, ids, props = csvutils.read_csv_with_header_and_index(properties_file)
        # change order of properties
        props = apply_unit_conversion(header, props)

        # read skip list
        skip_list = prepare_skip_list(skip_file)

        # last round of processing
        self.mols, self.props = self._prepare_data(
            mols, props, skip_list, 
        )

        # filter data if needed
        if self.pre_filter is not None:
            filtered_data = [d for d in zip(self.mols, self.props) if self.pre_filter(d)]
            self.mols, self.props = zip(*filtered_data)

        # apply pre_transform if needed
        if self.pre_transform is not None:
            transformed_data = [self.pre_transform(d) for d in zip(self.mols, self.props)]
            self.mols, self.props = zip(*transformed_data)

        # save data
        self.save(self.mols, self.raw_paths[0])
        self.save(self.props, self.raw_paths[1])

        # get statistics
        self.stats = molutils.get_molecule_stats(self.mols, self.atom_types_repr)
        self.atom_types = self.stats['atom_types']
        self.bond_types = self.stats['bond_types']
        self.charges = self.stats['charges']
        self.auxiliary_node_state_values = self.stats.get(AUXILIARY_NODE_STATES_STATS_KEY)
        
        # store data in files
        self.save(self.stats, self.raw_paths[2])


        
    def _prepare_data( self, mols, props, skip_list):

        final_mols, final_props = [], []

        skipped_skiplist = 0
        skipped_sanitization = 0

        progbar = tqdm(range(len(mols)), desc='Processing molecules')

        for i in progbar:

            progbar.set_postfix_str(
                f'Skipped: {skipped_skiplist} (skip list), {skipped_sanitization} (sanit. failed)',
                refresh= i % 1000 == 0
            )

            # get molecule and properties
            mol = mols[i]
            prop = props[i]

            if i in skip_list:  # skip if in skip list (read file '3195404' in the raws to know why)
                skipped_skiplist += 1
                continue
            if mol is None:     # skip if molecule is None (e.g., sanitization failed)
                skipped_sanitization += 1
                continue

            if self.remove_hydrogens:
                mol = molutils.remove_hydrogens_from_molecule(mol)

            if self.kekulize:
                mol = molutils.kekulize_molecule(mol)

            final_mols.append(mol)
            final_props.append(prop)
            
        return final_mols, final_props


    def __len__(self):
        return len(self.mols)
        
    def __getitem__(self, idx):
        return self.mols[idx], self.props[idx]
    

class QM9(MolecularGraphsDataset):

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
            pre_transform_raw=None,
            pre_filter_raw=None,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            atom_types_repr: str= 'default'
        ):

        if root is None:
            root = DEFAULT_DATASET_PATH_QM9

        # create raw dataset
        raw_dataset = QM9Raw(
            root, sanitize=sanitize,
            remove_hydrogens=remove_hydrogens, kekulize=kekulize,
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
        )

class QM9Smiles(SmilesDataset):
    
    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            sanitize: bool = True,
            remove_hydrogens: bool = True,
            kekulize: bool = True,
            pre_transform_raw=None,
            pre_filter_raw=None,
            pre_transform=None,
            pre_filter=None,
            atom_types_repr=None
        ):

        if root is None:
            root = DEFAULT_DATASET_PATH_QM9

        # create raw dataset
        raw_dataset = QM9Raw(
            root, sanitize=sanitize,
            remove_hydrogens=remove_hydrogens, kekulize=kekulize,
            pre_transform=pre_transform_raw, pre_filter=pre_filter_raw,
            atom_types_repr=atom_types_repr
        )

        super().__init__(
            root, split=split, raw_mol_dataset=raw_dataset,
            pre_transform=pre_transform, pre_filter=pre_filter
        )


from src.data.datasets.split import random_split_dataset
from src.data.datasets import reg_dataresources

@reg_dataresources.register('qm9')
class QM9Resources(DataResources):

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
            pre_transform=None,
            pre_filter=None,
            pre_transform_raw=None,
            pre_filter_raw=None,
            atom_types_repr: str = 'default',
        ):

        super().__init__()

        self.root = root
        
        self.qm9_cfg = {
            'sanitize': sanitize,
            'remove_hydrogens': remove_hydrogens,
            'kekulize': kekulize,
            'hard_remove_hydrogens': hard_remove_hydrogens,
            'include_pos': include_pos,
            'include_charges': include_charges,
            'pre_transform_raw': pre_transform_raw,
            'pre_filter_raw': pre_filter_raw,
            'atom_types_repr': atom_types_repr,
        }
        self.smiles_cfg = {
            'sanitize': sanitize,
            'remove_hydrogens': remove_hydrogens,
            'kekulize': kekulize
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

        ds = QM9(self.root, **self.qm9_cfg)
        ds_smiles = QM9Smiles(self.root, **self.smiles_cfg)

        self.decoder = ds.mol_to_torch_converter
        self.info_total = ds.stats

        # if there is any pre_transform, resolve any transform adapter
        self.preproc['pre_transform'] = self.transforms_to_pipeline(self.preproc['pre_transform'])
        self.preproc['pre_filter'] = self.filters_to_pipeline(self.preproc['pre_filter'])

        try: # try to get the split datasets

            dss = {split: [
                    QM9(self.root, split=split, **self.preproc, **self.qm9_cfg),
                    QM9Smiles(self.root, split=split, **self.smiles_cfg)
                ] for split in self.random_splits
            }

        except DatasetException: # if not possible, create the splits
            print('Creating random splits for QM9 graphs and SMILES')

            # reload dataset with preprocessing
            if self.preproc['pre_transform'] is not None or self.preproc['pre_filter'] is not None:
                print('Applying preprocessing to the whole dataset')
                ds.reapply_pre_transform(self.preproc['pre_transform'], self.preproc['pre_filter'])

            dss = random_split_dataset([ds, ds_smiles], self.random_splits)

        self.info = {split: self.wrap_dataset(d[0]).stats for split, d in dss.items()}

        self._prepared = True


    def get(self, resource: str=None, split: str=None, transform=None):
        if not self._prepared:
            self.prepare_data()

        if resource == 'dataset':
            return self.wrap_dataset(QM9(self.root, split=split, **self.preproc, **self.qm9_cfg), transform=transform)
        elif resource == 'smiles':
            return QM9Smiles(self.root, split=split, **self.smiles_cfg)
        elif resource == 'decoder':
            return self.decoder
        elif resource == 'info':
            return self.info[split] if split in self.info else self.info_total
        else:
            raise ValueError(f'Resource {resource} not found for QM9 dataset, choose between "dataset" and "smiles"')
        

    def __repr__(self):
        return f'{self.__class__.__name__}[resources=[dataset, smiles, decoder, info], splits={list(self.random_splits.keys())}]'
    

@reg_dataresources.register('qm9_mmff')
class QM9Resources(DataResources):

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
            pre_transform=None,
            pre_filter=None,
            pre_transform_raw=None,
            pre_filter_raw=None,
            atom_types_repr: str = 'default',
        ):

        super().__init__()
        
        self.root = root
        
        self.qm9_cfg = {
            'sanitize': sanitize,
            'remove_hydrogens': remove_hydrogens,
            'kekulize': kekulize,
            'hard_remove_hydrogens': hard_remove_hydrogens,
            'include_pos': include_pos,
            'include_charges': include_charges,
            'pre_transform_raw': pre_transform_raw,
            'pre_filter_raw': pre_filter_raw,
            'atom_types_repr': atom_types_repr,
        }
        self.smiles_cfg = {
            'sanitize': sanitize,
            'remove_hydrogens': remove_hydrogens,
            'kekulize': kekulize
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

        ds = QM9(self.root, **self.qm9_cfg)
        ds_smiles = QM9Smiles(self.root, **self.smiles_cfg)

        self.decoder = ds.mol_to_torch_converter
        self.info_total = ds.stats

        # if there is any pre_transform, resolve any transform adapter
        self.preproc['pre_transform'] = self.transforms_to_pipeline(self.preproc['pre_transform'])
        self.preproc['pre_filter'] = self.filters_to_pipeline(self.preproc['pre_filter'])

        try: # try to get the split datasets

            dss = {split: [
                    QM9(self.root, split=split, **self.preproc, **self.qm9_cfg),
                    QM9Smiles(self.root, split=split, **self.smiles_cfg)
                ] for split in self.random_splits
            }

        except DatasetException: # if not possible, create the splits
            print('Creating random splits for QM9 graphs and SMILES')

            # reload dataset with preprocessing
            if self.preproc['pre_transform'] is not None or self.preproc['pre_filter'] is not None:
                print('Applying preprocessing to the whole dataset')
                ds.reapply_pre_transform(self.preproc['pre_transform'], self.preproc['pre_filter'])

            dss = random_split_dataset([ds, ds_smiles], self.random_splits)

        self.info = {split: self.wrap_dataset(d[0]).stats for split, d in dss.items()}

        self._prepared = True


    def get(self, resource: str=None, split: str=None, transform=None):
        if not self._prepared:
            self.prepare_data()

        if resource == 'dataset':
            return self.wrap_dataset(QM9(self.root, split=split, **self.preproc, **self.qm9_cfg), transform=transform)
        elif resource == 'smiles':
            return QM9Smiles(self.root, split=split, **self.smiles_cfg)
        elif resource == 'decoder':
            return self.decoder
        elif resource == 'info':
            return self.info[split] if split in self.info else self.info_total
        else:
            raise ValueError(f'Resource {resource} not found for QM9 dataset, choose between "dataset" and "smiles"')
        

    def __repr__(self):
        return f'{self.__class__.__name__}[resources=[dataset, smiles, decoder, info], splits={list(self.random_splits.keys())}]'
    

@reg_dataresources.register('qm9_atom_details')
class QM9Resources(DataResources):

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
            pre_transform=None,
            pre_filter=None,
            pre_transform_raw=None,
            pre_filter_raw=None,
            atom_types_repr: str = 'default',
        ):

        super().__init__()
        
        self.root = root
        
        self.qm9_cfg = {
            'sanitize': sanitize,
            'remove_hydrogens': remove_hydrogens,
            'kekulize': kekulize,
            'hard_remove_hydrogens': hard_remove_hydrogens,
            'include_pos': include_pos,
            'include_charges': include_charges,
            'pre_transform_raw': pre_transform_raw,
            'pre_filter_raw': pre_filter_raw,
            'atom_types_repr': atom_types_repr,
        }
        self.smiles_cfg = {
            'sanitize': sanitize,
            'remove_hydrogens': remove_hydrogens,
            'kekulize': kekulize
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

        ds = QM9(self.root, **self.qm9_cfg)
        ds_smiles = QM9Smiles(self.root, **self.smiles_cfg)

        self.decoder = ds.mol_to_torch_converter
        self.info_total = ds.stats

        # if there is any pre_transform, resolve any transform adapter
        self.preproc['pre_transform'] = self.transforms_to_pipeline(self.preproc['pre_transform'])
        self.preproc['pre_filter'] = self.filters_to_pipeline(self.preproc['pre_filter'])

        try: # try to get the split datasets

            dss = {split: [
                    QM9(self.root, split=split, **self.preproc, **self.qm9_cfg),
                    QM9Smiles(self.root, split=split, **self.smiles_cfg)
                ] for split in self.random_splits
            }

        except DatasetException: # if not possible, create the splits
            print('Creating random splits for QM9 graphs and SMILES')

            # reload dataset with preprocessing
            if self.preproc['pre_transform'] is not None or self.preproc['pre_filter'] is not None:
                print('Applying preprocessing to the whole dataset')
                ds.reapply_pre_transform(self.preproc['pre_transform'], self.preproc['pre_filter'])

            dss = random_split_dataset([ds, ds_smiles], self.random_splits)

        self.info = {split: self.wrap_dataset(d[0]).stats for split, d in dss.items()}

        self._prepared = True


    def get(self, resource: str=None, split: str=None, transform=None):
        if not self._prepared:
            self.prepare_data()

        if resource == 'dataset':
            return self.wrap_dataset(QM9(self.root, split=split, **self.preproc, **self.qm9_cfg), transform=transform)
        elif resource == 'smiles':
            return QM9Smiles(self.root, split=split, **self.smiles_cfg)
        elif resource == 'decoder':
            return self.decoder
        elif resource == 'info':
            return self.info[split] if split in self.info else self.info_total
        else:
            raise ValueError(f'Resource {resource} not found for QM9 dataset, choose between "dataset" and "smiles"')
        

    def __repr__(self):
        return f'{self.__class__.__name__}[resources=[dataset, smiles, decoder, info], splits={list(self.random_splits.keys())}]'