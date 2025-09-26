from typing import Dict

from torch_geometric.transforms import BaseTransform
from src.datatypes.sparse import SparseGraph
from src.data.transforms.core import TransformAdapter

from src.datatypes.utils import one_hot

from src.data.transforms import reg_transforms

from src.data.simple_transforms.molecular import mol2smiles, smiles2mol


class SmilesNoChiralityTransform(BaseTransform):
    
    def __init__(
            self,
            remove_hydrogens: bool = True,
            kekulize: bool = True
        ):
        super().__init__()
        self.remove_hydrogens = remove_hydrogens
        self.kekulize = kekulize
    
    def forward(self, smiles: str) -> str:
        
        mol = smiles2mol(
            smiles,
            sanitize=False,
            remove_hydrogens=self.remove_hydrogens,
            kekulize=self.kekulize
        )
        return mol2smiles(mol, sanitize=False, isomeric=False)