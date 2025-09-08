from typing import List, Optional, Callable
from tqdm import tqdm

from collections import OrderedDict
from tqdm.contrib.concurrent import process_map

import os.path as osp

import torch
from torch_geometric.io.fs import makedirs
from src.data.datasets.core import RawDataset, ProcessedDataset

from src.data.simple_transforms.molecular import GraphToMoleculeConverter, mol2smiles, smiles2mol
from src.data.utils.graphs import get_torch_graphs_stats
from src.datatypes.sparse import SparseGraph

import src.data.utils.molecular as molutils


from copy import copy, deepcopy


class MolecularGraphsDataset(ProcessedDataset):
    """Here I'm using the InMemoryDataset class from PyTorch Geometric
    to be compatible
    """

    def __init__(
            self,
            root: str,
            raw_mol_dataset: RawDataset,
            atom_types: List[str],
            bond_types: List[str],
            charges: Optional[List[int]] = None,
            split: Optional[str] = None,
            hard_remove_hydrogens: bool = False,
            include_pos: bool = False,
            include_charges: bool = False,
            num_workers: int = 0,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None
        ) -> None:

        # hydrogen removal: during process method call all hydrogens are removed
        self.hard_remove_hydrogens = hard_remove_hydrogens
        self.include_pos = include_pos
        self.include_charges = include_charges
        self.num_workers = num_workers
        if hard_remove_hydrogens and 'H' in atom_types:
            atom_types.remove('H')

        # assign numbers to each atom and bond type
        atom_decoder = {atom: i for i, atom in enumerate(atom_types)}
        bond_decoder = {bond: i for i, bond in enumerate(bond_types)}
        if charges is not None:
            charge_decoder = {charge: i for i, charge in enumerate(charges)}

        self.mol_to_torch_converter = GraphToMoleculeConverter(
            atom_decoder = atom_decoder,
            bond_decoder = bond_decoder,
            charge_decoder = charge_decoder if charges is not None else None,
            include_pos = include_pos,
            include_charges = include_charges
        )

        self.raw_mol_dataset = raw_mol_dataset

        # call super constructor -> process data
        super().__init__(root, split, transform, pre_transform, pre_filter)

        # remove reference to base dataset, no need for it
        del self.raw_mol_dataset

        self.load(self.processed_paths[0], SparseGraph)
        self.stats = self.load_file(self.processed_paths[1])


    def subset_from(self, indices: List[int], name: str):

        subset = copy(self)
        subset.root = self.root
        subset.split = name
        makedirs(subset.processed_dir)
        subset.save([self[i] for i in indices], subset.processed_paths[0])
        subset.load(subset.processed_paths[0], SparseGraph)

        new_stats = get_torch_graphs_stats(subset)
        subset.stats = deepcopy(self.stats)
        subset.stats.update(new_stats)
        subset.save_file(subset.stats, subset.processed_paths[1])

        return subset


    @property
    def processed_file_names(self) -> str:
        return 'data.pt', 'stats.json'
    
    
    def _prepare_data_worker(self, data):
        # get molecule and properties
        molecule, properties = self.data_to_mol_and_prop(data)
        # convert molecule to graph (optional: fully remove hydrogens)
        graph = self.mol_to_torch_converter.molecule_to_graph(
            molecule, hard_remove_hydrogens=self.hard_remove_hydrogens
        )
        # add properties to graph if there are any
        if properties is not None:
            graph.y = torch.tensor(properties)
        
        return graph
        

    def process(self):
        
        if self.num_workers > 0 and False:
            chunksize = len(self.raw_mol_dataset) // self.num_workers // 5
            
            # transform
            results = process_map(
                self._prepare_data_worker, self.raw_mol_dataset,
                max_workers=self.num_workers, desc='Converting Chem.Mols to SparseGraphs', chunksize=chunksize,
            )
        else:
            # no parallelization, just convert
            results = [self._prepare_data_worker(data) for data in tqdm(self.raw_mol_dataset, desc='Converting Chem.Mols to SparseGraphs')]
        
        self.pre_transform_filter_and_finalize(results, self.pre_transform, self.pre_filter)


        
    def pre_transform_filter_and_finalize(
            self,
            input_graphs: List[SparseGraph],
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None
        ) -> None:
        # useful method to apply a pre_transform/pre_filter to an already processed dataset
        graphs = []
        for graph in input_graphs:

            # apply pre_transform if any
            if pre_filter is not None and not pre_filter(graph):
                continue
            if pre_transform is not None:
                graph = pre_transform(graph)

            graphs.append(graph)

        self.stats = {
            'num_cls_nodes': len(self.mol_to_torch_converter.atom_decoder),
            'num_cls_edges': len(self.mol_to_torch_converter.bond_decoder),
            'num_cls_properties': graphs[0].y.size(0) if hasattr(graphs[0], 'y') and graphs[0].y is not None else 0,
            **get_torch_graphs_stats(graphs)
        }
        if self.include_charges and self.mol_to_torch_converter.charge_decoder is not None:
            self.stats['num_cls_charges'] = len(self.mol_to_torch_converter.charge_decoder)
        
        self.save(graphs, self.processed_paths[0])
        self.save_file(self.stats, self.processed_paths[1])
        

    def reapply_pre_transform(self, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None):
        # add pre_transform and pre_filter to the dataset
        self.add_pre_transforms_filters(pre_transform, pre_filter)
        # apply new pre_transform and pre_filter to the dataset
        self.pre_transform_filter_and_finalize(self, pre_transform, pre_filter)
        # reload data
        self.load(self.processed_paths[0], SparseGraph)
        self.stats = self.load_file(self.processed_paths[1])


    def data_to_mol_and_prop(self, sample):
        if isinstance(sample, (tuple, list)):
            mol, props = sample
            if isinstance(props, dict):
                props = list(props.values())
            return mol, props
        else: # no properties
            return sample, None
        

class SmilesDataset(RawDataset):

    def __init__(
            self,
            root: str,
            raw_mol_dataset: RawDataset,
            split: Optional[str] = None,
            pre_transform=None,
            pre_filter=None
        ):

        self.raw_mol_dataset = raw_mol_dataset
        super().__init__(root, split=split, pre_transform=pre_transform, pre_filter=pre_filter)
        del self.raw_mol_dataset

        if not hasattr(self, 'smiles'):
            self.smiles = self.load(self.raw_paths[0])


    def subset_from(self, indices: List[int], name: str):

        subset = copy(self)
        subset.root = self.root
        subset.split = name
        makedirs(subset.raw_dir)
        subset.smiles = [self.smiles[i] for i in indices]
        subset.save(subset.smiles, subset.raw_paths[0])

        return subset

    
    @property
    def raw_file_names(self):
        return ['smiles.json']
    
    def download(self):

        # get all smiles from the raw dataset
        smiles = []
        for data in tqdm(self.raw_mol_dataset, desc='Converting Chem.Mols to SMILES strings'):
            # get molecule and properties
            molecule, properties = self.data_to_mol_and_prop(data)
            # convert molecule to smiles
            smiles.append(mol2smiles(molecule))

        self.smiles = smiles
        self.save(self.smiles, self.raw_paths[0])
    

    def data_to_mol_and_prop(self, sample):
        if isinstance(sample, (tuple, list)):
            mol, props = sample
            if isinstance(props, dict):
                props = list(props.values())
            return mol, props
        else: # no properties
            return sample, None
        

    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        return self.smiles[idx]
    

class MolecularDataset(RawDataset):

    def __init__(
            self,
            root: str,
            raw_smiles_dataset: RawDataset,
            split: Optional[str] = None,
            sanitize: bool = False,
            remove_hydrogens: bool = False,
            kekulize: bool = False,
            properties_computer_function: Optional[Callable] = None,
            pre_transform=None,
            pre_filter=None
        ):

        self.sanitize = sanitize
        self.remove_hydrogens = remove_hydrogens
        self.kekulize = kekulize
        self.properties_computer_function = properties_computer_function

        self.raw_smiles_dataset = raw_smiles_dataset
        super().__init__(root, split=split, pre_transform=pre_transform, pre_filter=pre_filter)
        del self.raw_smiles_dataset

        if not hasattr(self, 'mols'):
            self.mols = self.load(self.raw_paths[0])
            self.props = self.load(self.raw_paths[1])
            self.stats = self.load(self.raw_paths[2])
            self.atom_types = self.stats['atom_types']
            self.bond_types = self.stats['bond_types']


    def subset_from(self, indices: List[int], name: str):
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
        stats_new = molutils.get_molecule_stats(subset.mols)
        stats_new['atom_types'] = self.atom_types # use old atom types
        stats_new['bond_types'] = self.bond_types # use old bond types
        subset.stats = stats_new
        subset.atom_types = self.atom_types
        subset.bond_types = self.bond_types
        
        # store data in files
        subset.save(subset.stats, subset.raw_paths[2])

        return subset

    
    @property
    def raw_file_names(self):
        return ['mols.pkl', 'props_mols.json', 'stats.json']
    
    def download(self):

        # get all smiles from the raw dataset
        mols = []
        props = []
        for data in tqdm(self.raw_smiles_dataset, desc='Converting SMILES strings to Chem.Mols'):
            # get smiles and properties
            smiles, properties = self.data_to_smiles_and_prop(data)

            # convert molecule to smiles
            mol = smiles2mol(smiles, sanitize=self.sanitize, remove_hydrogens=self.remove_hydrogens, kekulize=self.kekulize)

            # if smiles conversion returned None, discard the molecule
            if mol is None:
                continue

            mols.append(mol)

            # compute properties if needed
            if properties is None or len(properties) == 0:
                properties = OrderedDict()

            if self.properties_computer_function is not None:
                properties.update(self.properties_computer_function(mol))

            props.append(properties)

        self.mols = mols
        self.props = props
        
        # save data
        self.save(self.mols, self.raw_paths[0])
        self.save(self.props, self.raw_paths[1])

        # get statistics
        self.stats = molutils.get_molecule_stats(self.mols)
        self.atom_types = self.stats['atom_types']
        self.bond_types = self.stats['bond_types']
        
        # store data in files
        self.save(self.stats, self.raw_paths[2])
    

    def data_to_smiles_and_prop(self, sample):
        if isinstance(sample, (tuple, list)):
            mol, props = sample
            if isinstance(props, dict):
                props = list(props.values())
            return mol, props
        else: # no properties
            return sample, None
        

    def __len__(self):
        return len(self.mols)
    
    def __getitem__(self, idx):
        return self.mols[idx], self.props[idx]