from typing import Dict, List, Tuple, Optional, Union
import re

import copy

import numpy as np

import torch
from torch import Tensor
from torch_geometric.utils import subgraph, to_undirected
from src.datatypes.sparse import SparseGraph

from src.datatypes.sparse import to_directed

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Geometry import Point3D
from rdkit import DistanceGeometry
from rdkit.Chem import AllChem
import networkx as nx

from src.data.simple_transforms import batched

BOND_TYPES_REAL = {1: BT.SINGLE, 2: BT.DOUBLE, 3: BT.TRIPLE, 4: BT.AROMATIC}
BOND_TYPES_REAL_REV = {v: k for k, v in BOND_TYPES_REAL.items()}
BOND_TYPES = {bt: str(bt) for bt in [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]}
BOND_TYPES_REV = {str(bt): bt for bt in [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]}
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

CHARGE_ENCODER = {0: 0, -1: 1, 1: 2}

@batched
def mol2smiles(mol, sanitize=True):
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
    smiles = Chem.MolToSmiles(mol, canonical=True)

    return smiles

@batched
def smiles2mol(smiles, sanitize=True, remove_hydrogens=False, kekulize=False):
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is not None:
        if remove_hydrogens:
            mol = Chem.RemoveHs(mol)
        if kekulize:
            Chem.Kekulize(mol)
    return mol


@batched
def mol2nx(mol):

    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                    label=atom.GetSymbol())
        #    atomic_num=atom.GetAtomicNum(),
        #    formal_charge=atom.GetFormalCharge(),
        #    chiral_tag=atom.GetChiralTag(),
        #    hybridization=atom.GetHybridization(),
        #    num_explicit_hs=atom.GetNumExplicitHs(),
        #    is_aromatic=atom.GetIsAromatic())

    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    label=int(bond.GetBondTypeAsDouble()))
        #    bond_type=bond.GetBondType())

    return G


def pos_from_dist_rdkit(dist_matrix):
    return DistanceGeometry.DistGeom.EmbedBoundsMatrix(dist_matrix.double().numpy())

def get_pos_from_mol(mol: Chem.Mol) -> Optional[Tensor]:
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        pos = torch.tensor(pos, dtype=torch.float)

    else: # compute positions from the molecule itself
        try:
            # get molecule
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            AllChem.UFFOptimizeMolecule(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 0xf00d
            mol=Chem.RemoveHs(mol)
            mol_block = Chem.MolToMolBlock(mol)
            
            coordinates = []
            lines = mol_block.splitlines()
            for line in lines[4:4 + mol.GetNumAtoms()]:  # The coordinates start at line 5
                parts = line.split()
                # Extract the x, y, z coordinates (3rd, 4th, and 5th columns)
                x, y, z = map(float, parts[0:3])
                for elem in [x,y,z]:
                    coordinates.append(elem)
            
            n=len(coordinates)

            pos = torch.tensor(coordinates, dtype=torch.float32).view(int(n/3),3)
            
        except Exception as e:
            print(f"Error in get_pos_from_mol: {e}")
            return None
    
    return pos - torch.mean(pos, dim=0, keepdim=True)


def build_molecule(
        atom_types: Tensor,
        edge_index: Tensor,
        edge_types: Tensor,
        atom_decoder: Union[Dict[int, str], Dict[str, int]],
        bond_decoder: Optional[Dict[int, str]]=None,
        charge_decoder: Optional[Union[Dict[int, float], Dict[float, int]]]=None,
        charges: Optional[Tensor]=None,
        pos: Optional[Tensor]=None,
        distance: Optional[Tensor]=None,
        relaxed: bool=False,
        verbose: bool=False
    ) -> Chem.Mol:
    if verbose:
        print("building new molecule")

    ###############################  PARSE ATOMS  ##############################
    mol = Chem.RWMol()
    for i, atom in enumerate(atom_types):
        a = Chem.Atom(atom_decoder[atom.item()])
        if charges is not None:
            charge = charge_decoder[charges[i].item()]
            if charge != 0:
                a.SetFormalCharge(charge)
                if verbose:
                    print("charge added: ", charge, atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])

    ###############################  PARSE BONDS  ##############################

    for bond, link in zip(edge_types, edge_index.permute(1, 0)):

        if link[0].item() != link[1].item():
            mol.AddBond(link[0].item(), link[1].item(), BOND_TYPES_REV[bond_decoder[bond.item()]])
            if verbose:
                print(
                    "bond added:", link[0].item(), link[1].item(), bond.item(),
                      bond_decoder[bond.item()]
                )

            if relaxed and charges is None:
                # add formal charge to atom: e.g. [O+], [N+], [S+]
                # not support [O-], [N-], [S-], [NH+] etc.
                flag, atomid_valence = check_valency(mol)
                if verbose:
                    print("flag, valence", flag, atomid_valence)
                if flag:
                    continue
                else:
                    assert len(atomid_valence) == 2
                    idx = atomid_valence[0]
                    v = atomid_valence[1]
                    an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                    if verbose:
                        print("atomic num of atom with a large valence", an)
                    if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                        mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                        # print("Formal charge added")
                        
    try:
        mol = mol.GetMol()
    except Chem.KekulizeException:
        print("Can't kekulize molecule")
        return None
                        
    if pos != None:
        positions = pos.double()
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
        mol.AddConformer(conf)
    
    if distance != None:
        try:
            pos = pos_from_dist_rdkit(distance.reshape(len(atom_types), len(atom_types)))
            pos= pos-np.mean(pos,axis=0)

            conf = Chem.Conformer(mol.GetNumAtoms())
            for i in range(mol.GetNumAtoms()):
                            conf.SetAtomPosition(i, Point3D(pos[i][0], pos[i][1], pos[i][2]))
            mol.AddConformer(conf)
        except Exception as e:
            pass

    return mol



def build_graph_from_molecule(
        mol: Chem.Mol,
        atom_encoder: Union[Dict[int, str], Dict[str, int]],
        bond_encoder: Union[Dict[int, str], Dict[str, int]],
        charge_encoder: Optional[Union[Dict[int, float], Dict[float, int]]] = None,
        hard_remove_hydrogens: bool=False,
        include_pos: bool=False,
        include_charges: bool=False
    ):

    if hard_remove_hydrogens:
        h_idx = len(atom_encoder)
        atom_encoder = {**atom_encoder, 'H': h_idx} # add temporary hydrogen to atom encoder
        
    if include_pos:
        pos = get_pos_from_mol(mol)
        
    if include_charges:
        charges = [charge_encoder[atom.GetFormalCharge()] for atom in mol.GetAtoms()]
        charges = torch.tensor(charges, dtype=torch.long)

    # build graph from molecule
    atom_types = []
    edge_index = []
    edge_types = []

    for atom in mol.GetAtoms():
        atom_types.append(atom_encoder[atom.GetSymbol()])
    

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = str(bond.GetBondType())

        edge_index.append([start, end])
        edge_types.append(bond_encoder[bond_type])
    

    # tranform to tensor
    x = torch.tensor(atom_types, dtype=torch.long)
    if len(edge_index) == 0: # if no edges, then set special case
        edge_index = torch.tensor([[], []], dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).permute(1, 0)
    edge_attr = torch.tensor(edge_types, dtype=torch.long)

    if hard_remove_hydrogens:
        to_keep = x < h_idx # keep all atoms that are not hydrogen
        edge_index, edge_attr = subgraph(
            to_keep, edge_index, edge_attr, relabel_nodes=True,
            num_nodes=len(to_keep)
        )
        x = x[to_keep]

    # make graph undirected
    edge_index, edge_attr = to_undirected(edge_index, edge_attr, num_nodes=x.size(0))
    
    addons = {}
    
    if include_pos:
        addons['node_pos'] = pos
    if include_charges:
        addons['node_charges'] = charges

    g = SparseGraph(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        **addons
    )

    return g


# from GDSS
def valid_mol_can_with_seg(x, largest_connected_comp=True):
    if x is None:
        return None
    sm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = Chem.MolFromSmiles(sm)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    return mol


class GraphToMoleculeConverter:

    def __init__(
            self,
            atom_decoder: Union[Dict[int, str], Dict[str, int]],
            bond_decoder: Union[Dict[int, str], Dict[str, int]],
            charge_decoder: Optional[Union[Dict[int, float], Dict[float, int]]]=None,
            relaxed: bool=False,
            post_hoc_mols_fix: bool=False,
            post_hoc_mols_convert: bool=False,
            include_pos: bool=False,
            include_charges: bool=False,
        ):
        
        self.relaxed = relaxed
        self.post_hoc_mols_fix = post_hoc_mols_fix
        self.post_hoc_mols_convert = post_hoc_mols_convert
        self.include_pos = include_pos
        self.include_charges = include_charges

        self.atom_decoder = check_decoder(atom_decoder, int)
        self.atom_encoder = check_decoder(atom_decoder, str)
        self.bond_decoder = check_decoder(bond_decoder, int)
        self.bond_encoder = check_decoder(bond_decoder, str)
        if charge_decoder is not None and self.include_charges:
            self.charge_decoder = check_decoder(charge_decoder, int)
            self.charge_encoder = check_decoder(charge_decoder, float)
        else:
            self.charge_decoder = None
            self.charge_encoder = None


    def __call__(
            self,
            batch: Union[List[SparseGraph], SparseGraph],
            override_relaxed: Optional[bool]=None,
            override_post_hoc_mols_fix: Optional[bool]=None,
            override_post_hoc_mols_convert: Optional[bool]=None
        ) -> Union[List[Chem.Mol], Chem.Mol]:

        is_batch = True

        if not isinstance(batch, list):
            # check and format for batch
            is_batch = hasattr(batch, 'ptr')

            if is_batch:
                batch = batch.to_data_list()
            else:
                batch = [batch]

        # build molecules from graphs
        out_molecules = []

        g: SparseGraph
        for g in batch:

            g = g.clone()

            if g.is_undirected():
                g.edge_index, g.edge_attr = to_directed(g.edge_index, g.edge_attr)

            # collapse classes if needed
            g.collapse('x', 'edge_attr', 'node_charges')

            mol = build_molecule(
                atom_types =		g.x,
                pos =               g.node_pos if self.include_pos else None,
                charges =           g.node_charges if self.include_charges else None,
                edge_index =		g.edge_index,
                edge_types =		g.edge_attr,
                atom_decoder =		self.atom_decoder,
                bond_decoder =		self.bond_decoder,
                charge_decoder =    self.charge_decoder,
                relaxed =		    self.relaxed if override_relaxed is None else override_relaxed
            )

            # apply correction if needed
            mol = self.correct_mol(mol, override_post_hoc_mols_fix)

            # apply conversion if needed
            mol = self.convert_mol_and_back(mol, override_post_hoc_mols_convert)

            out_molecules.append(mol)


        return out_molecules if is_batch else out_molecules[0]
    
    def graph_to_molecule(
            self,
            batch: Union[List[SparseGraph], SparseGraph],
            override_relaxed: Optional[bool]=None,
            override_post_hoc_mols_fix: Optional[bool]=None,
            override_post_hoc_mols_convert: Optional[bool]=None
        ) -> Union[List[Chem.Mol], Chem.Mol]:

        return self(
            batch,
            override_relaxed,
            override_post_hoc_mols_fix,
            override_post_hoc_mols_convert
        )
            

    def molecule_to_graph(
            self,
            mols: Union[List[Chem.Mol], List[str], Chem.Mol, str],
            kekulize: bool=True,
            hard_remove_hydrogens: bool=False,
            override_post_hoc_mols_fix: Optional[bool]=None,
            override_post_hoc_mols_convert: Optional[bool]=None
        ) -> SparseGraph:

        single_mol = False

        # check element
        if isinstance(mols, (str, Chem.Mol)):
            mols = [mols]
            single_mol = True

        out_graphs = []

        # start conversion
        for mol in mols:

            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)

            if kekulize:
                mol = copy.deepcopy(mol)
                Chem.Kekulize(mol)

            # apply correction if needed
            mol = self.correct_mol(mol, override_post_hoc_mols_fix)

            # apply conversion if needed
            mol = self.convert_mol_and_back(mol, override_post_hoc_mols_convert)

            # build graph from molecule
            g = build_graph_from_molecule(
                mol,
                atom_encoder = self.atom_encoder,
                bond_encoder = self.bond_encoder,
                charge_encoder= self.charge_encoder,
                hard_remove_hydrogens = hard_remove_hydrogens,
                include_pos = self.include_pos,
                include_charges = self.include_charges
            )

            out_graphs.append(g)

        return out_graphs[0] if single_mol else out_graphs
    

    
    def correct_mol(
            self,
            mol: Chem.Mol,
            override_post_hoc_mols_fix: Optional[bool]=None
        ):

        if override_post_hoc_mols_fix is None:
            if self.post_hoc_mols_fix:
                mol, _ = correct_mol(mol)

        else:
            if override_post_hoc_mols_fix:
                mol, _ = correct_mol(mol)

        return mol
    
    def convert_mol_and_back(
            self,
            mol: Chem.Mol,
            override_post_hoc_mols_convert: Optional[bool]=None
        ):

        if override_post_hoc_mols_convert is None:
            if self.post_hoc_mols_convert:
                mol = valid_mol_can_with_seg(mol)

        else:
            if override_post_hoc_mols_convert:
                mol = valid_mol_can_with_seg(mol)

        return mol



def check_decoder(decoder: Dict, type_key: type):
    """ Check if the decoder is in the correct format.
    If not, reverse the mapping.
    """
    if len(decoder) > 0 and not isinstance(next(iter(decoder.keys())), type_key):
        decoder = {v: k for k, v in decoder.items()}
    
    return decoder

def check_atom_bond_decoders(
        atom_decoder,
        bond_decoder,
        atom_idx_to_name: bool=True,
    ):

    if atom_idx_to_name:
        type_key = int
    else:
        type_key = str

    # if atom_idx_to_name is True:
    # if atom_decoder is given as atom_name -> atom_idx
    # reverse mapping to atom_idx -> atom_name
    # else: do the same with atom_idx -> atom_name
    if len(atom_decoder) > 0 and not isinstance(next(iter(atom_decoder.keys())), type_key):
        atom_decoder = {v: k for k, v in atom_decoder.items()}
    
    if len(bond_decoder) > 0 and not isinstance(next(iter(bond_decoder.keys())), type_key):
        bond_decoder = {v: k for k, v in bond_decoder.items()}

    return atom_decoder, bond_decoder


# Functions from GDSS
def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence
    


# code from: https://github.com/GRAPH-0/CDGS/blob/main/utils.py
def correct_mol(
        mol: Chem.Mol,
    ):

    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            queue = []

            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append(
                    (b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                )
            queue.sort(key=lambda tup: tup[1], reverse=True)

            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, BOND_TYPES_REAL[t])

    return mol, no_correct