import copy
from collections import OrderedDict
from pyparsing import ABC
from rdkit import Chem
from rdkit.Chem import ChemicalForceFields
import torch

from src.data.datasets import reg_atom_types_representation


AUXILIARY_NODE_STATES_STATS_KEY = 'auxiliary_node_state_values'

NODE_MMFF_TYPE_ATTR = 'node_mmff_type'
NODE_IS_IN_RING_ATTR = 'node_is_in_ring'
NODE_IS_AROMATIC_ATTR = 'node_is_aromatic'
NODE_HYBRIDIZATION_ATTR = 'node_hybridization'

def infer_auxiliary_node_state_values(stats: dict | None) -> OrderedDict[str, list]:
    if stats is None:
        return OrderedDict()

    explicit_values = stats.get(AUXILIARY_NODE_STATES_STATS_KEY)
    if explicit_values:
        return OrderedDict((name, list(values)) for name, values in explicit_values.items())

    return OrderedDict()


def infer_auxiliary_node_state_marginals(stats: dict | None, auxiliary_node_state_values: OrderedDict[str, list]) -> OrderedDict[str, list]:
    if stats is None or len(auxiliary_node_state_values) == 0:
        return OrderedDict()

    marginals = stats.get('marginals', {})
    if all(name in marginals for name in auxiliary_node_state_values):
        return OrderedDict((name, marginals[name]) for name in auxiliary_node_state_values)

    return OrderedDict()


class AtomTypeRepresentation(ABC):
    """Base class for atom representation."""

    auxiliary_node_attr_names = ()

    def __init__(self, molecule: Chem.Mol):
        super().__init__()
        self.molecule = molecule

    def get_atom_label(self, atom: Chem.rdchem.Atom) -> str:
        return atom.GetSymbol()

    def get_auxiliary_node_features(self, atom: Chem.rdchem.Atom) -> OrderedDict[str, object]:
        return OrderedDict()

    def has_auxiliary_representation(self) -> bool:
        return len(self.auxiliary_node_attr_names) > 0

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

    auxiliary_node_attr_names = (NODE_MMFF_TYPE_ATTR,)

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

    def get_auxiliary_node_features(self, atom: Chem.rdchem.Atom) -> OrderedDict[str, object]:
        return OrderedDict([
            (NODE_MMFF_TYPE_ATTR, str(self.mmff_props.GetMMFFAtomType(atom.GetIdx())))
        ])

    def encode_atom_representation(self, atom: Chem.rdchem.Atom) -> str:
        return self.get_atom_label(atom)

    def decode_atom_representation(self, atom: str) -> str:
        return atom.split("_")[0]
    

@reg_atom_types_representation.register("atom_details")
class AtomDetailsRepresentation(AtomTypeRepresentation):
    """Base class for atom representation."""

    auxiliary_node_attr_names = (
        NODE_IS_IN_RING_ATTR,
        NODE_IS_AROMATIC_ATTR,
        NODE_HYBRIDIZATION_ATTR,
    )

    def __init__(self, molecule: Chem.Mol):
        super().__init__(molecule)
        self.ordered_keys = ["In Ring", "Aromatic", "Hybridization", "Formal Charge"]

    def get_auxiliary_node_features(self, atom: Chem.rdchem.Atom) -> OrderedDict[str, object]:
        return OrderedDict([
            (NODE_IS_IN_RING_ATTR, atom.IsInRing()),
            (NODE_IS_AROMATIC_ATTR, atom.GetIsAromatic()),
            (NODE_HYBRIDIZATION_ATTR, atom.GetHybridization().name),
        ])

    def encode_atom_representation(self, atom: Chem.rdchem.Atom) -> str:
        return self.get_atom_label(atom)
        

    def decode_atom_representation(self, atom: str) -> str:
        return atom.split("_")[0]
