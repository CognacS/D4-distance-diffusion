from typing import List, Optional, Callable
from tqdm import tqdm

from collections import OrderedDict
from tqdm.contrib.concurrent import process_map

import os.path as osp

import torch
from torch_geometric.io.fs import makedirs
from src.data.datasets.core import RawDataset, ProcessedDataset

from src.data.simple_transforms.molecular import GraphToMoleculeConverter, mol2smiles, smiles2mol, mol2mol
from src.data.utils.graphs import get_torch_graphs_stats
from src.datatypes.sparse import SparseGraph
from src.data.simple_transforms.molecular import verify_and_compute_3d_conformer
from src.data.datasets.atom_types_representation import AUXILIARY_NODE_STATES_STATS_KEY

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
            chunksize: Optional[int] = None,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            not_splittable: bool = False,
            atom_types_repr: str = 'default'
        ) -> None:

        # hydrogen removal: during process method call all hydrogens are removed
        self.hard_remove_hydrogens = hard_remove_hydrogens
        self.include_pos = include_pos
        self.include_charges = include_charges
        self.num_workers = num_workers
        self.chunksize = chunksize
        self.atom_types_repr = atom_types_repr
        if hard_remove_hydrogens and 'H' in atom_types:
            atom_types.remove('H')

        # assign numbers to each atom and bond type
        atom_decoder = {atom: i for i, atom in enumerate(atom_types)}
        bond_decoder = {bond: i for i, bond in enumerate(bond_types)}
        auxiliary_node_state_values = getattr(raw_mol_dataset, 'auxiliary_node_state_values', None)
        if auxiliary_node_state_values is None and hasattr(raw_mol_dataset, 'stats'):
            auxiliary_node_state_values = raw_mol_dataset.stats.get(AUXILIARY_NODE_STATES_STATS_KEY)
        if auxiliary_node_state_values is not None:
            auxiliary_node_state_decoders = {
                name: {value: i for i, value in enumerate(values)}
                for name, values in auxiliary_node_state_values.items()
            }
        else:
            auxiliary_node_state_decoders = None
        if charges is not None:
            charge_decoder = {charge: i for i, charge in enumerate(charges)}

        self.auxiliary_node_state_values = auxiliary_node_state_values

        self.mol_to_torch_converter = GraphToMoleculeConverter(
            atom_decoder = atom_decoder,
            bond_decoder = bond_decoder,
            auxiliary_node_state_decoders = auxiliary_node_state_decoders,
            charge_decoder = charge_decoder if charges is not None else None,
            include_pos = include_pos,
            include_charges = include_charges,
            atom_types_repr = atom_types_repr
        )

        self.raw_mol_dataset = raw_mol_dataset

        # call super constructor -> process data
        super().__init__(root, split, transform, pre_transform, pre_filter, not_splittable=not_splittable)

        # remove reference to base dataset, no need for it
        del self.raw_mol_dataset

        self.load(self.processed_paths[0], SparseGraph)
        self.stats = self.load_file(self.processed_paths[1])


    def subset_from(self, indices: List[int], name: str):

        subset = copy(self)
        subset.root = self.root
        subset.split = name
        makedirs(subset.processed_dir)
        subset.save([self[i] for i in indices if i < len(self)], subset.processed_paths[0])
        subset.load(subset.processed_paths[0], SparseGraph)

        num_cls = {
            'x': len(self.mol_to_torch_converter.atom_decoder),
            'edge_attr': len(self.mol_to_torch_converter.bond_decoder),
        }
        for name, decoder in self.mol_to_torch_converter.auxiliary_node_state_decoders.items():
            num_cls[name] = len(decoder)
        if self.include_charges and self.mol_to_torch_converter.charge_decoder is not None:
            num_cls['node_charges'] = len(self.mol_to_torch_converter.charge_decoder)

        new_stats = get_torch_graphs_stats(subset, num_classes=num_cls)
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
            molecule, hard_remove_hydrogens=self.hard_remove_hydrogens,
            kekulize=False, atom_types_repr=self.atom_types_repr # kekulization will be done before if needed
        )
        # add properties to graph if there are any
        if properties is not None:
            graph.y = torch.tensor(properties)
        
        return graph
        

    def process(self):
        
        dataset = [self.raw_mol_dataset[i] for i in range(len(self.raw_mol_dataset))]
        #dataset = [self.raw_mol_dataset[i] for i in range(10000)] # TEMPORARY, REMOVE LATER

        if self.num_workers > 0:
            if self.chunksize is not None:
                chunksize = self.chunksize
            else:
                chunksize = len(dataset) // self.num_workers // 5
            
            # transform
            results = process_map(
                self._prepare_data_worker, dataset,
                max_workers=self.num_workers, desc='Converting Chem.Mols to SparseGraphs', chunksize=chunksize,
            )
        else:
            # no parallelization, just convert
           #results = [self._prepare_data_worker(data) for data in tqdm(self.raw_mol_dataset, desc='Converting Chem.Mols to SparseGraphs')]
           results = []
           for data in tqdm(dataset, desc='Converting Chem.Mols to SparseGraphs'):
               results.append(self._prepare_data_worker(data))
            # results = []
            # for data in dataset:
            #     results.append(self._prepare_data_worker(data))
        
        self.pre_transform_filter_and_finalize(results, self.pre_transform, self.pre_filter)


        
    def pre_transform_filter_and_finalize(
            self,
            input_graphs: List[SparseGraph],
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None
        ) -> None:
        # useful method to apply a pre_transform/pre_filter to an already processed dataset
        graphs = []
        for i, graph in enumerate(input_graphs):

            # apply pre_transform if any
            if pre_filter is not None and not pre_filter(graph, i):
                continue
            if pre_transform is not None:
                graph = pre_transform(graph)

            graphs.append(graph)
            
        num_cls = {
            'x': len(self.mol_to_torch_converter.atom_decoder),
            'edge_attr': len(self.mol_to_torch_converter.bond_decoder),
        }
        for name, decoder in self.mol_to_torch_converter.auxiliary_node_state_decoders.items():
            num_cls[name] = len(decoder)

        self.stats = {
            'num_cls_nodes': num_cls['x'],
            'num_cls_edges': num_cls['edge_attr'],
            'num_cls_properties': graphs[0].y.size(0) if hasattr(graphs[0], 'y') and graphs[0].y is not None else 0
        }
        if self.auxiliary_node_state_values is not None:
            self.stats[AUXILIARY_NODE_STATES_STATS_KEY] = self.auxiliary_node_state_values
            for name, values in self.auxiliary_node_state_values.items():
                self.stats[f'num_cls_{name}'] = len(values)
        if self.include_charges and self.mol_to_torch_converter.charge_decoder is not None:
            num_cls['node_charges'] = len(self.mol_to_torch_converter.charge_decoder)
            self.stats['num_cls_charges'] = num_cls['node_charges']
        
        self.stats.update(get_torch_graphs_stats(graphs, num_classes=num_cls))
        
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
            pre_filter=None,
            not_splittable: bool = False
        ):

        self.raw_mol_dataset = raw_mol_dataset
        super().__init__(root, split=split, pre_transform=pre_transform, pre_filter=pre_filter, not_splittable=not_splittable)
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
        for i, data in enumerate(tqdm(self.raw_mol_dataset, desc='Converting Chem.Mols to SMILES strings')):
            
            if self.pre_filter is not None and not self.pre_filter(data, i):
                continue
                
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
            pre_filter=None,
            atom_types_repr: str='default'
        ):

        self.sanitize = sanitize
        self.remove_hydrogens = remove_hydrogens
        self.kekulize = kekulize
        self.properties_computer_function = properties_computer_function
        self.atom_types_repr = atom_types_repr

        self.raw_smiles_dataset = raw_smiles_dataset
        super().__init__(root, split=split, pre_transform=pre_transform, pre_filter=pre_filter)
        del self.raw_smiles_dataset

        if not hasattr(self, 'mols'):
            self.mols = self.load(self.raw_paths[0])
            self.props = self.load(self.raw_paths[1])
            self.stats = self.load(self.raw_paths[2])
            self.atom_types = self.stats['atom_types']
            self.bond_types = self.stats['bond_types']
            self.charges = self.stats['charges'] if 'charges' in self.stats else None
            self.auxiliary_node_state_values = self.stats.get(AUXILIARY_NODE_STATES_STATS_KEY)


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
        self.stats = molutils.get_molecule_stats(self.mols, self.atom_types_repr)
        self.atom_types = self.stats['atom_types']
        self.bond_types = self.stats['bond_types']
        self.charges = self.stats['charges']
        self.auxiliary_node_state_values = self.stats.get(AUXILIARY_NODE_STATES_STATS_KEY)
        
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


class MoleculeDatasetWorker:
    
    def __init__(
            self,
            sanitize: bool = False,
            remove_hydrogens: bool = False,
            kekulize: bool = False,
            compute_3d_conformer: bool = False,
            properties_computer_function: Optional[Callable] = None,
            pre_transform=None,
            pre_filter=None
        ):
        self.sanitize = sanitize
        self.remove_hydrogens = remove_hydrogens
        self.kekulize = kekulize
        self.compute_3d_conformer = compute_3d_conformer
        self.properties_computer_function = properties_computer_function
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
    
    
    def __call__(self, args) -> tuple:
        
        smiles_or_mol: str
        props: OrderedDict
        smiles_or_mol, props = args
        
        try:
            if isinstance(smiles_or_mol, str):
                # convert smiles to mol
                mol = smiles2mol(smiles_or_mol, sanitize=self.sanitize, remove_hydrogens=self.remove_hydrogens)
            else:
                # preprocess mol
                mol = mol2mol(smiles_or_mol, sanitize=self.sanitize, remove_hydrogens=self.remove_hydrogens)

            # if needed, compute 3D conformer
            if self.compute_3d_conformer and mol is not None:
                mol = verify_and_compute_3d_conformer(mol)
                
            if self.kekulize and mol is not None:
                mol = molutils.kekulize_molecule(mol)
            
            # if something went wrong, discard the molecule
            if mol is None:
                return (None, None, None)

            if self.pre_filter is not None and not self.pre_filter(mol):
                return (None, None, None)

            if self.pre_transform is not None:
                mol = self.pre_transform(mol)

            if self.properties_computer_function is not None:
                p_new = self.properties_computer_function(mol)
                props.update(p_new) # add new properties to the existing ones
                
            return (mol, props, None)
        
        except Exception as e:
            # print(f'Error processing molecule: {e}')
            return (None, None, str(e))
    
class ExtendedMolecularDatasetRaw(RawDataset):

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
            not_splittable: bool = False,
            atom_types_repr: str='default'
        ):
        
        self.sanitize = sanitize
        self.remove_hydrogens = remove_hydrogens
        self.kekulize = kekulize
        self.properties_computer_function = properties_computer_function
        self.compute_3d_conformer = compute_3d_conformer
        self.num_workers = num_workers
        self.chunksize = chunksize
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.atom_types_repr = atom_types_repr

        super().__init__(root, split=split, pre_transform=pre_transform, pre_filter=pre_filter, not_splittable=not_splittable)

    
    
    def load_data(self, mols_path: str, props_path: str, stats_path: str):
        if not hasattr(self, 'mols'):
            self.mols = self.load(mols_path)
            self.props = self.load(props_path)
            self.stats = self.load(stats_path)
            self.atom_types = self.stats['atom_types']
            self.bond_types = self.stats['bond_types']
            self.charges = self.stats['charges'] if 'charges' in self.stats else None
            self.auxiliary_node_state_values = self.stats.get(AUXILIARY_NODE_STATES_STATS_KEY)


    def subset_from(self, indices: List[int], name: str, mols_path: str, props_path: str, stats_path: str):
        """Create a subset of the dataset with the indices provided, at the folder
        name provided. This is useful for splitting the dataset into train, test, and validation
        """

        subset = copy(self)
        subset.root = self.root
        subset.split = name
        makedirs(subset.raw_dir)
        subset.mols = [self.mols[i] for i in indices if i < len(self.mols)]
        subset.props = [self.props[i] for i in indices if i < len(self.props)]

        # save data
        subset.save(subset.mols, mols_path)
        subset.save(subset.props, props_path)

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
        subset.save(subset.stats, stats_path)

        return subset


    def preprocess_molecules(self, smiles_or_mols: List[str], props: List[OrderedDict], mols_path: str, props_path: str) -> tuple:

        type_of_data = 'SMILES' if isinstance(smiles_or_mols[0], str) else 'raw Chem.Mols'

        ###################  LOAD DATA IF IT ALREADY EXISTS  ###################
        # if already exists mols and props in files, load them
        if osp.exists(mols_path) and osp.exists(props_path):
            print('Molecules and properties files already exist, loading them...')
            mols = self.load(mols_path)
            props = self.load(props_path)
            print(f'Loaded {len(mols)} molecules from files.',
                  'If this should not be the case, please delete the files and re-run.')

        else:
            ####################  PREPARE LISTS AND WORKER  ####################
            mols = []
            new_props = []
            how_many = len(smiles_or_mols)
            iterable = zip(smiles_or_mols, props)
            worker = MoleculeDatasetWorker(
                sanitize=self.sanitize,
                remove_hydrogens=self.remove_hydrogens,
                kekulize=self.kekulize,
                compute_3d_conformer=self.compute_3d_conformer,
                properties_computer_function=self.properties_computer_function,
                pre_transform=self.pre_transform,
                pre_filter=self.pre_filter
            )

            ########################  MULTIPROCESSING  #########################
            if self.num_workers > 0:
                if self.chunksize is None:
                    chunksize = max(1, how_many // (20 * self.num_workers))
                else:
                    chunksize = self.chunksize
                
                results = process_map(
                    worker, iterable, max_workers=self.num_workers, chunksize=chunksize,
                    desc=f'Converting {type_of_data} to processed Chem.Mols', total=how_many
                )
                for mol, p, err in results:
                    if err is None and mol is not None:
                        mols.append(mol)
                        new_props.append(p)
                    else:
                        print(err)
            
            #######################  NO MULTIPROCESSING  #######################
            else:
                for s_or_m, p in tqdm(iterable, total=how_many, desc=f'Converting {type_of_data} to processed Chem.Mols'):
                    mol, p, err = worker((s_or_m, p))
                    if err is None and mol is not None:
                        mols.append(mol)
                        new_props.append(p)
                    else:
                        print(err)
            
            # save data
            self.save(mols, mols_path)
            self.save(new_props, props_path)

        return mols, new_props


    def __len__(self):
        return len(self.mols)
    
    def __getitem__(self, idx):
        return self.mols[idx], self.props[idx]