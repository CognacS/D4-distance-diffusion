from pyparsing import ABC
from rdkit import Chem
from rdkit.Chem import ChemicalForceFields

from src.data.datasets import reg_atom_types_representation


class AtomTypeRepresentation(ABC):
    """Base class for atom representation."""

    def __init__(self, molecule: Chem.Mol):
        super().__init__()
        self.molecule = molecule

    def encode_atom_representation(self, atom: Chem.rdchem.Atom) -> str:
        raise NotImplementedError

    def decode_atom_representation(self, atom: str) -> str:
        raise NotImplementedError


@reg_atom_types_representation.register("default")
class DefaultAtomTypeRepresentation(AtomTypeRepresentation):
    """Default class for atom representation."""

    def __init__(self, molecule: Chem.Mol):
        super().__init__(molecule)

    def encode_atom_representation(self, atom: Chem.rdchem.Atom) -> str:
        return atom.GetSymbol()

    def decode_atom_representation(self, atom: str) -> str:
        return atom


@reg_atom_types_representation.register("MMFF")
class MMFFAtomTypeRepresentation(AtomTypeRepresentation):
    """Base class for atom representation."""

    def __init__(self, molecule: Chem.Mol):
        super().__init__(molecule)

        # creating force field properties
        self.mmff_props = ChemicalForceFields.MMFFGetMoleculeProperties(self.molecule)

    def encode_atom_representation(self, atom: Chem.rdchem.Atom) -> str:
        return f"{atom.GetSymbol()}_{self.mmff_props.GetMMFFAtomType(atom.GetIdx())}"

    def decode_atom_representation(self, atom: str) -> str:
        return atom.split("_")[0]
