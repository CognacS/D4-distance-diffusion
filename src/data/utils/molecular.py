from typing import List, Optional
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalForceFields
from rdkit.Chem.rdchem import BondType as bt
from tqdm import tqdm
import copy

from src.data.simple_transforms.molecular import BOND_TYPES_REAL_REV, BOND_TYPES_REV
from src.data.datasets import reg_atom_types_representation
from src.data.datasets.atom_types_representation import AUXILIARY_NODE_STATES_STATS_KEY


RDLogger.DisableLog('rdApp.*')

import numpy as np


def read_molecules(filepath: str, sanitize: bool=False, remove_hydrogens: bool=False):
    """ Read molecules from a file.
    """

    ############  molecules file  ############
    if filepath.endswith('.sdf'):

        suppl = Chem.SDMolSupplier(
            filepath,
            removeHs=remove_hydrogens, # warning: this does not actually remove hydrogens
            sanitize=sanitize
        )

    #############  smiles file  ##############
    elif filepath.endswith('.smi'):
        
        suppl = Chem.SmilesMolSupplier(
            filepath,
            sanitize=sanitize
        )

    else:
        raise NotImplementedError(
            f'Molecules supplier for file {filepath} not implemented'
        )

    return suppl


def kekulize_molecule(mol):
    """ Kekulize a molecule.
    """
    Chem.Kekulize(mol)

    return mol

def remove_hydrogens_from_molecule(mol):
    """ Remove hydrogens from a molecule.
    """
    return Chem.RemoveHs(mol)


def get_molecule_stats(mols: List[Chem.Mol], atom_types_repr: str= 'default'):
    """Function for computing general statistics on set of molecules.
    Currently returns:
    - number of atoms: avg, std, and total
    - number of bonds: avg, std, and total
    - a list with all found atom types
    - a list with all found bond types

    Parameters
    ----------
    mols : 
    """
    
    atoms = set()
    auxiliary_node_state_values = {}
    bonds = set()
    charges = set()
    l_num_atoms = []
    l_num_bonds = []

    for mol in tqdm(mols, desc='Computing molecules stats'):
        # counting number of atoms and bonds
        l_num_atoms.append(mol.GetNumAtoms())
        l_num_bonds.append(mol.GetNumBonds())

        # get mol encoder
        mol_encoder = reg_atom_types_representation.get_instance(atom_types_repr, molecule=mol)

        for atom in mol.GetAtoms():
            atoms.add(mol_encoder.get_atom_label(atom))
            auxiliary_features = mol_encoder.get_auxiliary_node_features(atom)
            for name, value in auxiliary_features.items():
                auxiliary_node_state_values.setdefault(name, set()).add(value)
            charges.add(float(atom.GetFormalCharge()))

        for bond in mol.GetBonds():
            bonds.add(str(bond.GetBondType()))

    atoms = sorted(list(atoms))
    bonds = list(bonds)
    # reorder bonds to match BOND_TYPES_REAl order
    # BOND_TYPES_REV maps from string to rdkit BondType
    # BOND_TYPES_REAL_REV maps from rdkit BondType to int
    bonds = sorted(bonds, key=lambda x: BOND_TYPES_REAL_REV[BOND_TYPES_REV[x]])
    charges = list(charges)
    # reorder charges, such that negative charges come first
    charges = sorted(charges)

    ret_dict = {
        'num_atoms_avg': np.mean(l_num_atoms).item(),
        'num_atoms_std': np.std(l_num_atoms).item(),
        'num_atoms_total': np.sum(l_num_atoms).item(),
        'num_bonds_avg': np.mean(l_num_bonds).item(),
        'num_bonds_std': np.std(l_num_bonds).item(),
        'num_bonds_total': np.sum(l_num_bonds).item(),
        'atom_types': atoms,
        'bond_types': bonds,
        'charges': charges
    }

    if len(auxiliary_node_state_values) > 0:
        ret_dict[AUXILIARY_NODE_STATES_STATS_KEY] = {
            name: sorted(list(values), key=lambda value: (not str(value).lstrip('-').isdigit(), int(value) if str(value).lstrip('-').isdigit() else str(value)))
            for name, values in auxiliary_node_state_values.items()
        }

    return ret_dict