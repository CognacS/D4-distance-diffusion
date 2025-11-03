from typing import List
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalForceFields
from rdkit.Chem.rdchem import BondType as bt
from tqdm import tqdm
import copy

from src.data.simple_transforms.molecular import BOND_TYPES_REAL_REV, BOND_TYPES_REV

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
    return Chem.RemoveHs(mol, sanitize=True)

# def get_molecule_stats(mols: List[Chem.Mol]):
#     """Function for computing general statistics on set of molecules.
#     Currently returns:
#     - number of atoms: avg, std, and total
#     - number of bonds: avg, std, and total
#     - a list with all found atom types
#     - a list with all found bond types

#     Parameters
#     ----------
#     mols : 
#     """
    
#     atoms = set()
#     bonds = set()
#     charges = set()
#     l_num_atoms = []
#     l_num_bonds = []

#     for mol in tqdm(mols, desc='Computing molecules stats'):
#         l_num_atoms.append(mol.GetNumAtoms())
#         l_num_bonds.append(mol.GetNumBonds())

#         for atom in mol.GetAtoms():
#             atoms.add(atom.GetSymbol())
#             charges.add(float(atom.GetFormalCharge()))

#         for bond in mol.GetBonds():
#             bonds.add(str(bond.GetBondType()))

#     atoms = sorted(list(atoms))
#     bonds = list(bonds)
#     # reorder bonds to match BOND_TYPES_REAl order
#     # BOND_TYPES_REV maps from string to rdkit BondType
#     # BOND_TYPES_REAL_REV maps from rdkit BondType to int
#     bonds = sorted(bonds, key=lambda x: BOND_TYPES_REAL_REV[BOND_TYPES_REV[x]])
#     charges = list(charges)
#     # reorder charges, such that negative charges come first
#     charges = sorted(charges)

#     ret_dict = {
#         'num_atoms_avg': np.mean(l_num_atoms).item(),
#         'num_atoms_std': np.std(l_num_atoms).item(),
#         'num_atoms_total': np.sum(l_num_atoms).item(),
#         'num_bonds_avg': np.mean(l_num_bonds).item(),
#         'num_bonds_std': np.std(l_num_bonds).item(),
#         'num_bonds_total': np.sum(l_num_bonds).item(),
#         'atom_types': atoms,
#         'bond_types': bonds,
#         'charges': charges
#     }

#     return ret_dict


def get_molecule_stats(mols: List[Chem.Mol]):
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
    bonds = set()
    charges = set()
    l_num_atoms = []
    l_num_bonds = []

    for mol in tqdm(mols, desc='Computing molecules stats'):
        l_num_atoms.append(mol.GetNumAtoms())
        l_num_bonds.append(mol.GetNumBonds())

        mp = ChemicalForceFields.MMFFGetMoleculeProperties(mol)
        # mp = ChemicalForceFields.MMFFGetMoleculeProperties(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))

        for atom in mol.GetAtoms():
            FFMM_atom_type = mp.GetMMFFAtomType(atom.GetIdx())
            atom_label = f"{atom.GetSymbol()}_{FFMM_atom_type}"
            atoms.add(atom_label)
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

    return ret_dict