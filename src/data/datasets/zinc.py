from typing import Optional, Callable, Dict, List
import os.path as osp

from src.data.datasets.dig_datasets import BaseDigSmilesRaw, BaseDigResources, DEFAULT_DATASET_PATH

from src.data.datasets.molecular import MolecularDataset, MolecularGraphsDataset

DEFAULT_DATASET_PATH_ZINC_DIG = osp.join(DEFAULT_DATASET_PATH, 'zinc')

class ZincSmiles(BaseDigSmilesRaw):

    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            return_only_smiles: bool = False,
            pre_transform=None,
            pre_filter=None
        ):

        if root is None:
            root = DEFAULT_DATASET_PATH_ZINC_DIG

        super().__init__(
            root=root,
            which_dataset='zinc250k',
            split=split,
            return_only_smiles=return_only_smiles,
            pre_transform=pre_transform,
            pre_filter=pre_filter
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


class ZincMolecules(MolecularDataset):
    
        def __init__(
                self,
                root: Optional[str] = None,
                split: Optional[str] = None,
                sanitize: bool = True,
                remove_hydrogens: bool = True,
                kekulize: bool = True,
                properties_computer_function: Optional[Callable] = None,
                pre_transform=None,
                pre_filter=None
            ):

            if root is None:
                root = DEFAULT_DATASET_PATH_ZINC_DIG

            raw_smiles_dataset = ZincSmiles(
                root,
                split=split,
                pre_transform=pre_transform,
                pre_filter=pre_filter
            )
    
            super().__init__(
                root=root,
                split=split,
                raw_smiles_dataset=raw_smiles_dataset,
                sanitize=sanitize,
                remove_hydrogens=remove_hydrogens,
                kekulize=kekulize,
                properties_computer_function=properties_computer_function
            )


class Zinc(MolecularGraphsDataset):
        
    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            sanitize: bool = True,
            remove_hydrogens: bool = True,
            kekulize: bool = True,
            hard_remove_hydrogens: bool = True,
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
            properties_computer_function=properties_computer_function,
            pre_transform=pre_transform_raw, pre_filter=pre_filter_raw
        )

        super().__init__(
            root, split=split, raw_mol_dataset=raw_dataset,
            atom_types=raw_dataset.atom_types, bond_types=raw_dataset.bond_types,
            hard_remove_hydrogens=hard_remove_hydrogens,
            transform=transform, pre_transform=pre_transform, pre_filter=pre_filter
        )


from src.data.datasets import reg_dataresources

@reg_dataresources.register('zinc250k')
class ZincResources(BaseDigResources):

    def __init__(
            self,
            random_splits: Dict,
            root: Optional[str] = None,
            sanitize: bool = True,
            remove_hydrogens: bool = True,
            kekulize: bool = True,
            hard_remove_hydrogens: bool = True,
            pre_transform=None,
            pre_filter=None
        ):
        
        qm9_cfg = {
            'sanitize': sanitize,
            'remove_hydrogens': remove_hydrogens,
            'kekulize': kekulize,
            'hard_remove_hydrogens': hard_remove_hydrogens
        }
        smiles_cfg = {} # here in case this is needed in the future

        super().__init__(
            root=root,
            random_splits=random_splits,
            dataset_cfg=qm9_cfg,
            smiles_cfg=smiles_cfg,
            dataset_cls=Zinc,
            smiles_cls=ZincSmiles,
            pre_transform=pre_transform,
            pre_filter=pre_filter
        )
        