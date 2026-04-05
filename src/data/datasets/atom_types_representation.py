import copy
from pyparsing import ABC
from rdkit import Chem
from rdkit.Chem import ChemicalForceFields

from src.data.datasets import reg_atom_types_representation


ATOM_TYPES_REPR_NODE_ATTR = 'node_atom_types_repr'
ATOM_TYPES_REPR_STATS_KEY = 'atom_types_repr_values'


class AtomTypeRepresentation(ABC):
    """Base class for atom representation."""

    auxiliary_node_attr_name = None
    auxiliary_stats_key = None

    def __init__(self, molecule: Chem.Mol):
        super().__init__()
        self.molecule = molecule

    def get_atom_label(self, atom: Chem.rdchem.Atom) -> str:
        return atom.GetSymbol()

    def get_auxiliary_representation(self, atom: Chem.rdchem.Atom) -> str | None:
        return None

    def has_auxiliary_representation(self) -> bool:
        return self.auxiliary_node_attr_name is not None and self.auxiliary_stats_key is not None

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
        return self.get_atom_label(atom)

    def decode_atom_representation(self, atom: str) -> str:
        return atom


@reg_atom_types_representation.register("MMFF")
class MMFFAtomTypeRepresentation(AtomTypeRepresentation):
    """Base class for atom representation."""

    auxiliary_node_attr_name = ATOM_TYPES_REPR_NODE_ATTR
    auxiliary_stats_key = ATOM_TYPES_REPR_STATS_KEY

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

    def get_auxiliary_representation(self, atom: Chem.rdchem.Atom) -> str:
        return str(self.mmff_props.GetMMFFAtomType(atom.GetIdx()))

    def encode_atom_representation(self, atom: Chem.rdchem.Atom) -> str:
        return f"{self.get_atom_label(atom)}_{self.get_auxiliary_representation(atom)}"

    def decode_atom_representation(self, atom: str) -> str:
        return atom.split("_")[0]
    

@reg_atom_types_representation.register("atom_details")
class AtomDetailsRepresentation(AtomTypeRepresentation):
    """Base class for atom representation."""

    auxiliary_node_attr_name = ATOM_TYPES_REPR_NODE_ATTR
    auxiliary_stats_key = ATOM_TYPES_REPR_STATS_KEY

    def __init__(self, molecule: Chem.Mol):
        super().__init__(molecule)
        self.ordered_keys = ["In Ring", "Aromatic", "Hybridization", "Formal Charge"]

    def get_auxiliary_representation(self, atom: Chem.rdchem.Atom) -> str:
        info = {
            "In Ring": atom.IsInRing(),
            "Aromatic": atom.GetIsAromatic(),
            "Hybridization": atom.GetHybridization().name,
            "Formal Charge": atom.GetFormalCharge(),
        }
        ordered_values = [info[key] for key in self.ordered_keys]
        return "_".join([str(value) for value in ordered_values])

    def encode_atom_representation(self, atom: Chem.rdchem.Atom) -> str:
        return f"{self.get_atom_label(atom)}_{self.get_auxiliary_representation(atom)}"
        

    def decode_atom_representation(self, atom: str) -> str:
        return atom.split("_")[0]
