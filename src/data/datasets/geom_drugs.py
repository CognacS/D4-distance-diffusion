from typing import List, Optional
import os.path as osp
from tqdm import tqdm


from torch.utils.data import Dataset

import src.data.utils.molecular as molutils
from torch_geometric.data import download_url

from src.data.datasets.core import RawDataset, DataResources, DatasetException, DEFAULT_DATASET_PATH, DEFAULT_SPLITS
from src.data.datasets.molecular import MolecularGraphsDataset, SmilesDataset
from src.data.datasets.atom_types_representation import ATOM_TYPES_REPR_STATS_KEY
from src.data.utils.storing import load_file


DEFAULT_DATASET_PATH_GEOM_DRUGS = osp.join(DEFAULT_DATASET_PATH, 'geom-drugs')

DATASETS_URLS = {
    'train': 'https://bits.csb.pitt.edu/files/geom_raw/train_data.pickle',
    'valid': 'https://bits.csb.pitt.edu/files/geom_raw/val_data.pickle',
    'test': 'https://bits.csb.pitt.edu/files/geom_raw/test_data.pickle'
}


class GeomDrugsRaw(RawDataset):
    """ Geom-Drugs dataset class for raw data.
    """

    def __init__(
            self,
            root: Optional[str] = None,
            split: str = 'train',
            sanitize: bool = False,
            remove_hydrogens: bool = True,
            kekulize: bool = True,
            pre_transform=None,
            pre_filter=None,
            atom_types_repr: str = 'default',
        ):

        self.sanitize = sanitize
        self.remove_hydrogens = remove_hydrogens
        self.kekulize = kekulize
        self.atom_types_repr = atom_types_repr

        assert split is not None, 'Split must be specified for Geom-Drugs dataset'

        if root is None:
            root = DEFAULT_DATASET_PATH_GEOM_DRUGS

        super().__init__(root, split=split, pre_transform=pre_transform, pre_filter=pre_filter, not_splittable=True)

        if not hasattr(self, 'mols'):
            self.mols = self.load(self.raw_paths[0])
            self.stats = self.load(self.raw_paths[1])
            self.atom_types = self.stats['atom_types']
            self.bond_types = self.stats['bond_types']
            self.charges = self.stats['charges'] if 'charges' in self.stats else None
            self.atom_types_repr_values = self.stats.get(ATOM_TYPES_REPR_STATS_KEY)


    
    def subset_from(self, indices: List[int], name: str) -> Dataset:
        raise DatasetException('Geom-Drugs cannot be split, as it already has predefined splits.')

    
    @property
    def raw_file_names(self):
        return ['mols.pkl', 'stats.json']
    
    @property
    def other_file_names(self):
        l = super().other_file_names
        return l + [self.split + '_data.pickle']


    def download(self):
        # download data from URLs
        mols_file = download_url(DATASETS_URLS[self.split], self.raw_dir)

        # right now, molecules have all hydrogens, and contain smiles
        all_data = load_file(mols_file, load_method='pickle')
        
        data_list = []
        for i, data in enumerate(tqdm(all_data)):
            smiles, all_conformers = data
            for j, mol in enumerate(all_conformers):
                if j >= 5:
                    break

                if self.kekulize:
                    mol = molutils.kekulize_molecule(mol)
                    
                if self.remove_hydrogens:
                    mol = molutils.remove_hydrogens_from_molecule(mol)

                if self.pre_filter is not None and not self.pre_filter(mol):
                    continue
                if self.pre_transform is not None:
                    mol = self.pre_transform(mol)

                data_list.append(mol)
                
        self.mols = data_list

        # save data
        self.save(self.mols, self.raw_paths[0])

        # get statistics
        self.stats = molutils.get_molecule_stats(self.mols, self.atom_types_repr)
        self.atom_types = self.stats['atom_types']
        self.bond_types = self.stats['bond_types']
        self.charges = self.stats['charges']
        self.atom_types_repr_values = self.stats.get(ATOM_TYPES_REPR_STATS_KEY)
        
        # store data in files
        self.save(self.stats, self.raw_paths[1])


    def __len__(self):
        return len(self.mols)
        
    def __getitem__(self, idx):
        return self.mols[idx]
    

class GeomDrugs(MolecularGraphsDataset):

    def __init__(
            self,
            root: Optional[str] = None,
            split: str = 'train',
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
            atom_types_repr: str = 'default',
        ):

        if root is None:
            root = DEFAULT_DATASET_PATH_GEOM_DRUGS

        # create raw dataset
        raw_dataset = GeomDrugsRaw(
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
            not_splittable=True, atom_types_repr=atom_types_repr
        )

class GeomDrugsSmiles(SmilesDataset):
    
    def __init__(
            self,
            root: Optional[str] = None,
            split: str = 'train',
            sanitize: bool = True,
            remove_hydrogens: bool = True,
            kekulize: bool = True,
            pre_transform_raw=None,
            pre_filter_raw=None,
            pre_transform=None,
            pre_filter=None
        ):

        if root is None:
            root = DEFAULT_DATASET_PATH_GEOM_DRUGS

        # create raw dataset
        raw_dataset = GeomDrugsRaw(
            root, sanitize=sanitize,
            remove_hydrogens=remove_hydrogens, kekulize=kekulize,
            pre_transform=pre_transform_raw, pre_filter=pre_filter_raw
        )

        super().__init__(
            root, split=split, raw_mol_dataset=raw_dataset,
            pre_transform=pre_transform, pre_filter=pre_filter,
            not_splittable=True
        )


from src.data.datasets.split import random_split_dataset
from src.data.datasets import reg_dataresources

@reg_dataresources.register('geom-drugs')
class GeomDrugsResources(DataResources):

    def __init__(
            self,
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

        self.geom_drugs_cfg = {
            'sanitize': sanitize,
            'remove_hydrogens': remove_hydrogens,
            'kekulize': kekulize,
            'hard_remove_hydrogens': hard_remove_hydrogens,
            'include_pos': include_pos,
            'include_charges': include_charges,
            'pre_transform_raw': pre_transform_raw,
            'pre_filter_raw': pre_filter_raw,
            'atom_types_repr': atom_types_repr
        }
        self.smiles_cfg = {
            'sanitize': sanitize,
            'remove_hydrogens': remove_hydrogens,
            'kekulize': kekulize
        }

        self.preproc = {
            'pre_transform': pre_transform,
            'pre_filter': pre_filter
        }

        self._prepared = False


    def prepare_data(self):

        ds = GeomDrugs(self.root, **self.geom_drugs_cfg)

        self.decoder = ds.mol_to_torch_converter
        self.info_total = ds.stats

        # if there is any pre_transform, resolve any transform adapter
        self.preproc['pre_transform'] = self.transforms_to_pipeline(self.preproc['pre_transform'])
        self.preproc['pre_filter'] = self.filters_to_pipeline(self.preproc['pre_filter'])

        # geom-drugs is already split
        dss = {split: [
                GeomDrugs(self.root, split=split, **self.preproc, **self.geom_drugs_cfg),
                GeomDrugsSmiles(self.root, split=split, **self.smiles_cfg)
            ] for split in ['train', 'valid', 'test']
        }

        # if the training dataset never had preprocessing applied, apply it now
        if dss['train'][0].x.ndim == 1:
            # reload dataset with preprocessing
            if self.preproc['pre_transform'] is not None or self.preproc['pre_filter'] is not None:
                print('Applying preprocessing to the training set...')
                dss['train'][0].reapply_pre_transform(self.preproc['pre_transform'], self.preproc['pre_filter'])


        self.info = {split: self.wrap_dataset(d[0]).stats for split, d in dss.items()}

        self._prepared = True


    def get(self, resource: str=None, split: str=None, transform=None):
        if not self._prepared:
            self.prepare_data()

        if resource == 'dataset':
            return self.wrap_dataset(GeomDrugs(self.root, split=split, **self.preproc, **self.geom_drugs_cfg), transform=transform)
        elif resource == 'smiles':
            return GeomDrugsSmiles(self.root, split=split, **self.smiles_cfg)
        elif resource == 'decoder':
            return self.decoder
        elif resource == 'info':
            return self.info[split] if split in self.info else self.info_total
        else:
            raise ValueError(f'Resource {resource} not found for GeomDrugs dataset, choose between "dataset" and "smiles"')


    def __repr__(self):
        return f'{self.__class__.__name__}[resources=[dataset, smiles, decoder, info], splits=[train, valid, test]]'