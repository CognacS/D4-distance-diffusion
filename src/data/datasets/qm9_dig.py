from typing import Optional, Callable, Dict, List
import os.path as osp

from src.data.datasets.dig_datasets import BaseDigSmilesRaw, BaseDigResources, DEFAULT_DATASET_PATH

from src.data.datasets.molecular import MolecularDataset, MolecularGraphsDataset

DEFAULT_DATASET_PATH_QM9_DIG = osp.join(DEFAULT_DATASET_PATH, 'qm9-dig')

class QM9DigSmiles(BaseDigSmilesRaw):

    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            return_only_smiles: bool = False,
            pre_transform=None,
            pre_filter=None
        ):

        if root is None:
            root = DEFAULT_DATASET_PATH_QM9_DIG

        super().__init__(
            root=root,
            which_dataset='qm9',
            split=split,
            return_only_smiles=return_only_smiles,
            pre_transform=pre_transform,
            pre_filter=pre_filter
        )


    def process_csv(self, header, ids, rows):
        """In QM9, the header is as follows:
        - 1st and 2nd columns are SMILES1 and SMILES2
        - the rest are properties: A, B, C, etc...
        """
        smiles = []
        props = []
        for row in rows:
            row_copy = row.copy()
            smiles.append(row_copy.pop('SMILES1'))
            row_copy.pop('SMILES2')
            props.append(row_copy)

        return smiles, props
    
    def process_test_indices(self, struct) -> List[int]:
        idxs = [int(i) for i in struct['valid_idxs']]
        return idxs


class QM9DigMolecules(MolecularDataset):
    
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
                root = DEFAULT_DATASET_PATH_QM9_DIG

            raw_smiles_dataset = QM9DigSmiles(
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


class QM9Dig(MolecularGraphsDataset):
        
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
            root = DEFAULT_DATASET_PATH_QM9_DIG

        raw_dataset = QM9DigMolecules(
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

@reg_dataresources.register('qm9_dig')
class QM9DigResources(BaseDigResources):

    def __init__(
            self,
            random_splits: Dict,
            root: Optional[str] = None,
            sanitize: bool = True,
            remove_hydrogens: bool = True,
            kekulize: bool = True,
            hard_remove_hydrogens: bool = True,
            pre_transform=None,
            pre_filter=None,
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
            dataset_cls=QM9Dig,
            smiles_cls=QM9DigSmiles,
            pre_transform=pre_transform,
            pre_filter=pre_filter
        )
        