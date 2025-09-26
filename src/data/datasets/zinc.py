from typing import Optional, Callable, Dict, List
import os.path as osp

from src.data.datasets.dig_datasets import BaseDigSmilesRaw, BaseDigMoleculesRaw, BaseDigResources, DEFAULT_DATASET_PATH

from src.data.datasets.molecular import MolecularDataset, MolecularGraphsDataset, SmilesDataset

DEFAULT_DATASET_PATH_ZINC_DIG = osp.join(DEFAULT_DATASET_PATH, 'zinc')
            
class ZincMolecules(BaseDigMoleculesRaw):
    
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
            pre_filter=None
        ):

        if root is None:
            root = DEFAULT_DATASET_PATH_ZINC_DIG

        super().__init__(
            which_dataset='zinc250k',
            root=root,
            split=split,
            sanitize=sanitize,
            remove_hydrogens=remove_hydrogens,
            kekulize=kekulize,
            compute_3d_conformer=compute_3d_conformer,
            properties_computer_function=properties_computer_function,
            num_workers=num_workers,
            chunksize=chunksize,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        
    def process_csv(self, header, ids, rows):
        """In Zinc, the header is as follows:
        - 1st is smiles
        - the rest are properties: logP, qed, SAS
        """
        smiles = []
        props = []
        for row in rows:
            row_copy = row.copy()
            smiles.append(row_copy.pop('smiles'))
            props.append(row_copy)

        return smiles, props
    
    def process_test_indices(self, struct) -> List[int]:
        return struct


class Zinc(MolecularGraphsDataset):
        
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
            pre_filter=None
        ):

        if root is None:
            root = DEFAULT_DATASET_PATH_ZINC_DIG

        raw_dataset = ZincMolecules(
            root=root, sanitize=sanitize,
            remove_hydrogens=remove_hydrogens, kekulize=kekulize,
            compute_3d_conformer=include_pos,
            properties_computer_function=properties_computer_function,
            num_workers=num_workers, chunksize=chunksize,
            pre_transform=pre_transform_raw, pre_filter=pre_filter_raw
        )

        super().__init__(
            root, split=split, raw_mol_dataset=raw_dataset,
            atom_types=raw_dataset.atom_types, bond_types=raw_dataset.bond_types,
            charges=raw_dataset.charges,
            hard_remove_hydrogens=hard_remove_hydrogens,
            include_pos=include_pos, include_charges=include_charges,
            transform=transform, pre_transform=pre_transform, pre_filter=pre_filter,
            num_workers=num_workers, chunksize=chunksize
        )


class ZincSmiles(SmilesDataset):

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
            root = DEFAULT_DATASET_PATH_ZINC_DIG
            
        # create raw dataset
        raw_dataset = ZincMolecules(
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


from src.data.datasets import reg_dataresources

@reg_dataresources.register('zinc250k')
class ZincResources(BaseDigResources):

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
            pre_filter_raw=None
        ):
        
        zinc_cfg = {
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
        }
        smiles_cfg = {
            'sanitize': sanitize,
            'remove_hydrogens': remove_hydrogens,
            'kekulize': kekulize,
            'include_pos': include_pos,
            'pre_transform_raw': pre_transform_raw,
            'pre_filter_raw': pre_filter_raw
        }

        super().__init__(
            root=root,
            random_splits=random_splits,
            dataset_cfg=zinc_cfg,
            smiles_cfg=smiles_cfg,
            dataset_cls=Zinc,
            smiles_cls=ZincSmiles,
            pre_transform=pre_transform,
            pre_filter=pre_filter
        )
        