import copy
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

        # creating force field properties. 
        # NOTE: This function modify in place the molecule calling the sanitization function!
        #       For this reason a deepcopy of the molecule should be passed to this function
        self.deepcopy_molecule = copy.deepcopy(self.molecule)
        self.mmff_props = ChemicalForceFields.MMFFGetMoleculeProperties(self.deepcopy_molecule)
        if self.mmff_props is None:
            print("Original SMILES:", Chem.MolToSmiles(self.molecule))
            raise ValueError("MMFF properties could not be computed for the given molecule.")

    def encode_atom_representation(self, atom: Chem.rdchem.Atom) -> str:
        return f"{atom.GetSymbol()}_{self.mmff_props.GetMMFFAtomType(atom.GetIdx())}"

    def decode_atom_representation(self, atom: str) -> str:
        return atom.split("_")[0]
    

@reg_atom_types_representation.register("atom_details")
class MMFFAtomTypeRepresentation(AtomTypeRepresentation):
    """Base class for atom representation."""

    def __init__(self, molecule: Chem.Mol):
        super().__init__(molecule)
        self.ordered_keys = ["Element", "In Ring", "Aromatic", "Hybridization", "Formal Charge"]

    def encode_atom_representation(self, atom: Chem.rdchem.Atom) -> str:
        info = {
            # "Index": atom.GetIdx(),
            "Element": atom.GetSymbol(),
            "In Ring": atom.IsInRing(),                     # se l'atomo fa parte di un anello
            "Aromatic": atom.GetIsAromatic(),               # se l'atomo fa parte di un anello aromatico
            "Hybridization": atom.GetHybridization().name,  # impatta il numero di legami che ha e potrebbe fare
            "Formal Charge": atom.GetFormalCharge(),        # carica formale dell'atomo
        }
        ordered_values = [info[key] for key in self.ordered_keys]
        string = "_".join([str(value) for value in ordered_values])
        return string
        

    def decode_atom_representation(self, atom: str) -> str:
        return atom.split("_")[0]
