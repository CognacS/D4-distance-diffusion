#!/usr/bin/env python3
"""Compare generated chemistry across representations and available training seeds.

Examples (run from the repository root)::

    python compare_generated_distributions.py
    python compare_generated_distributions.py --reference-split train
    python compare_generated_distributions.py --generated mmff=checkpoints/qm9-d4-mmff-v2

Every run is evaluated separately against the same cached reference. RDKit
recomputes all chemical descriptors, including MMFF and atom_details, for every
representation. Input datasets and checkpoints are only read.
"""

from argparse import ArgumentParser
from collections import Counter, OrderedDict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import gc
import hashlib
import io
import json
import re
import pickle
import statistics
import sys
import yaml
import math
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, rdchem, rdmolfiles
from tqdm.auto import tqdm

from torch_geometric.data import InMemoryDataset
from src.datatypes.sparse import SparseGraph
from src.data.datasets.atom_types_representation import (
    NODE_HYBRIDIZATION_ATTR,
    NODE_IS_AROMATIC_ATTR,
    NODE_IS_IN_RING_ATTR,
    NODE_MMFF_TYPE_ATTR,
)
from src.data.simple_transforms.molecular import mol2smiles


RDLogger.DisableLog("rdApp.*")

# suppress warnings from torch 2.4.1, aligned with the project entrypoints
warnings.filterwarnings("ignore", r"You are using .*torch.load.*weights_only=False.*")
warnings.filterwarnings("ignore", r".*deterministic implementation.*")
warnings.filterwarnings("ignore", r"Weights only load failed.*torch.load.*weights_only=True.*")
warnings.filterwarnings("ignore", r".*pre_transform.*pre-processed version of this dataset.*")
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*to-Python converter for.*")


DATASET_NAME_ALIASES = {
    "zinc": "zinc250k",
    "zinc_mmff": "zinc250k_mmff",
    "zinc_atom_details": "zinc250k_atom_details",
}

DATASET_ROOT_CANDIDATES = {
    "qm9": ["datasets/qm9"],
    "qm9_mmff": ["datasets/qm9_mmff"],
    "qm9_atom_details": ["datasets/qm9_atom_details"],
    "gdb13": ["datasets/gdb13"],
    "gdb13_mmff": ["datasets/gdb13_mmff"],
    "gdb13_atom_details": ["datasets/gdb13_atom_details"],
    "zinc250k": ["datasets/zinc250k", "datasets/zinc"],
    "zinc250k_mmff": ["datasets/zinc250k_mmff", "datasets/zinc_mmff"],
    "zinc250k_atom_details": ["datasets/zinc250k_atom_details", "datasets/zinc_atom_details"],
}

GENERATED_FILE_PATTERNS = (
    "generated_graphs*.pkl",
    "generated_graphs*.pt",
)

PLOT_CATEGORY_LIMIT = 20

SUBSTRUCTURE_SMARTS = {
    "benzene_ring": "c1ccccc1",
    "carbonyl": "[CX3]=[OX1]",
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "ester": "[#6][CX3](=O)[OX2][#6]",
    "ether": "[OD2]([#6])[#6]",
    "alcohol": "[OX2H][#6]",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
    "nitrile": "[CX2]#N",
    "halogen": "[F,Cl,Br,I]",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
}

SUBSTRUCTURE_PATTERNS = {
    name: rdmolfiles.MolFromSmarts(smarts)
    for name, smarts in SUBSTRUCTURE_SMARTS.items()
}


def parse_args() -> Any:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dataset", default="qm9")
    parser.add_argument("--reference-root", default=None)
    parser.add_argument("--reference-split", default="test",
                        help="test (default), train, valid/validation/val, or full")
    parser.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--model", choices=["d4", "midi"], default="d4")
    parser.add_argument("--representations", nargs="+", choices=["base", "mmff", "atom_details"],
                        default=["base", "mmff", "atom_details"])
    parser.add_argument("--generated", action="append",
                        help="Repeat NAME=PATH for explicit experiment folders or files. "
                             "NAME is base, mmff, or atom_details. A bare path uses --reference-dataset to infer NAME.")
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=None, help="JSON report path; plots are saved beside it")
    parser.add_argument("--max-reference", type=int, default=None)
    parser.add_argument("--max-generated", type=int, default=None, help="Sample cap PER training seed")
    parser.add_argument("--seed", type=int, default=0, help="Subsampling RNG seed (not the training seed)")
    parser.add_argument("--plot-top-k", type=int, default=PLOT_CATEGORY_LIMIT)
    args = parser.parse_args()
    for name in ["max_reference", "max_generated", "plot_top_k"]:
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if len(set(args.expected_seeds)) != len(args.expected_seeds):
        parser.error("--expected-seeds must be distinct")
    return args


def normalize_split(split: Optional[str]) -> Optional[str]:
    if split is None or split.lower() in {"none", "all", "full", "entire"}:
        return None
    aliases = {"val": "valid", "validation": "valid", "training": "train"}
    result = aliases.get(split.lower(), split.lower())
    if result not in {"train", "valid", "test"}:
        raise ValueError(f"Unsupported reference split: {split}")
    return result


def infer_atom_types_repr(dataset_name: str) -> str:
    if dataset_name.endswith("_mmff"):
        return "MMFF"
    if dataset_name.endswith("_atom_details"):
        return "atom_details"
    return "default"


def resolve_dataset_name(name: str) -> str:
    return DATASET_NAME_ALIASES.get(name, name)


def resolve_dataset_root(dataset_name: str, root_override: Optional[str]) -> str:
    if root_override is not None:
        return root_override

    candidates = DATASET_ROOT_CANDIDATES.get(dataset_name, [f"datasets/{dataset_name}"])
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return candidates[0]


class CompactGraphUnpickler(pickle._Unpickler):
    """Intern repeated tensor payloads before pickle memoizes them.

    Some generated files serialize the whole batch storage once per graph view.
    A normal pickle.load retains both all these bytes and all decoded storages.
    Override the Python unpickler's byte opcodes to share identical immutable
    payloads, and restore identical tensor storages once on CPU. This loader is
    for read-only analysis: the returned graphs must not be modified in place.
    """

    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(self, stream):
        super().__init__(stream)
        self.payloads = OrderedDict()
        self.storages = OrderedDict()
        self.payload_bytes = 0
        self.storage_bytes = 0
        self.cache_limit = 128 * 1024 * 1024

    def load_bytes(self, width):
        length = int.from_bytes(self.read(width), "little")
        if length > sys.maxsize:
            raise pickle.UnpicklingError("Byte payload exceeds platform size")
        value = self.read(length)
        if len(value) != length:
            raise pickle.UnpicklingError("Truncated byte payload")
        digest = hashlib.sha256(value).digest()
        if digest in self.payloads:
            value = self.payloads[digest]
            self.payloads.move_to_end(digest)
        elif length <= self.cache_limit:
            self.payloads[digest] = value
            self.payload_bytes += length
            while self.payload_bytes > self.cache_limit:
                _, removed = self.payloads.popitem(last=False)
                self.payload_bytes -= len(removed)
        self.append(value)

    def load_binbytes(self):
        self.load_bytes(4)

    def load_binbytes8(self):
        self.load_bytes(8)

    def load_short_binbytes(self):
        self.load_bytes(1)

    dispatch[pickle.BINBYTES[0]] = load_binbytes
    dispatch[pickle.BINBYTES8[0]] = load_binbytes8
    dispatch[pickle.SHORT_BINBYTES[0]] = load_short_binbytes

    def restore_storage(self, payload):
        digest = hashlib.sha256(payload).digest()
        if digest in self.storages:
            self.storages.move_to_end(digest)
            return self.storages[digest][0]
        storage = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        if len(payload) <= self.cache_limit:
            self.storages[digest] = (storage, len(payload))
            self.storage_bytes += len(payload)
            while self.storage_bytes > self.cache_limit:
                _, (_, removed_size) = self.storages.popitem(last=False)
                self.storage_bytes -= removed_size
        return storage

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return self.restore_storage
        return super().find_class(module, name)


def load_graph_collection(path: Path) -> List[Any]:
    if path.suffix == ".pkl":
        with path.open("rb") as stream:
            graphs = CompactGraphUnpickler(stream).load()
    else:
        graphs = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(graphs, list):
        return graphs
    if hasattr(graphs, "to_data_list"):
        return graphs.to_data_list()
    raise TypeError(f"Unsupported generated graph container: {type(graphs)!r}")


def maybe_subsample(items: Sequence[Any], limit: Optional[int], seed: int) -> List[Any]:
    items = list(items)
    if limit is None or limit >= len(items):
        return items
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(items), generator=generator)[:limit].tolist()
    return [items[index] for index in indices]


def select_indices(total_size: int, limit: Optional[int], seed: int) -> List[int]:
    if limit is None or limit >= total_size:
        return list(range(total_size))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator)[:limit].tolist()
    indices.sort()
    return indices


def collapse_classes(values: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if values is None:
        return None
    if values.ndim <= 1:
        return values.long()
    return values.argmax(dim=-1).long()


def update_counter(counter: Counter, values: Iterable[str]) -> None:
    for value in values:
        counter[str(value)] += 1


def update_nested_counters(counters: Dict[str, Counter], values_by_attr: Dict[str, Iterable[str]]) -> None:
    for attr_name, values in values_by_attr.items():
        update_counter(counters.setdefault(attr_name, Counter()), values)


def normalize_counter(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in sorted(counter.items())}


def normalize_nested_counters(counters: Dict[str, Counter]) -> Dict[str, Dict[str, float]]:
    return {name: normalize_counter(counter) for name, counter in sorted(counters.items())}


def total_variation_distance(counter_a: Counter, counter_b: Counter) -> Optional[float]:
    if sum(counter_a.values()) == 0 or sum(counter_b.values()) == 0:
        return None
    dist_a = normalize_counter(counter_a)
    dist_b = normalize_counter(counter_b)
    keys = set(dist_a) | set(dist_b)
    return 0.5 * sum(abs(dist_a.get(key, 0.0) - dist_b.get(key, 0.0)) for key in keys)


def nested_tvd(counters_a: Dict[str, Counter], counters_b: Dict[str, Counter]) -> Dict[str, Optional[float]]:
    return {
        attr_name: total_variation_distance(counters_a.get(attr_name, Counter()), counters_b.get(attr_name, Counter()))
        for attr_name in sorted(set(counters_a) | set(counters_b))
    }


def discrete_wasserstein(values_a: List[int], values_b: List[int]) -> Optional[float]:
    if not values_a or not values_b:
        return None

    counter_a = Counter(values_a)
    counter_b = Counter(values_b)
    support = sorted(set(counter_a) | set(counter_b))

    total_a = sum(counter_a.values())
    total_b = sum(counter_b.values())
    cumulative_a = 0.0
    cumulative_b = 0.0
    distance = 0.0
    previous_value: Optional[int] = None

    for value in support:
        if previous_value is not None:
            distance += abs(cumulative_a - cumulative_b) * (value - previous_value)
        cumulative_a += counter_a.get(value, 0) / total_a
        cumulative_b += counter_b.get(value, 0) / total_b
        previous_value = value

    return distance


def scalar_summary(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def round_floats(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: round_floats(item, digits=digits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item, digits=digits) for item in value]
    return value


def sanitize_plot_filename(name: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in name)


def infer_num_nodes(graph: Any) -> int:
    num_nodes = getattr(graph, "num_nodes", None)
    if num_nodes is not None:
        return int(num_nodes)
    return int(graph.x.shape[0])


def unique_undirected_edges(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> List[Tuple[Tuple[int, int], int]]:
    seen: Dict[Tuple[int, int], int] = {}
    if edge_attr is None:
        edge_attr = torch.zeros(edge_index.shape[1], dtype=torch.long)
    for pair, edge_value in zip(edge_index.t().tolist(), edge_attr.tolist()):
        src, dst = int(pair[0]), int(pair[1])
        key = (src, dst) if src <= dst else (dst, src)
        if key[0] == key[1]:
            continue
        if key not in seen:
            seen[key] = int(edge_value)
    return sorted(seen.items())


def count_connected_components(num_nodes: int, edges: Iterable[Tuple[Tuple[int, int], int]]) -> int:
    if num_nodes == 0:
        return 0

    adjacency = {node: set() for node in range(num_nodes)}
    for (src, dst), _ in edges:
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    visited = set()
    components = 0
    for node in range(num_nodes):
        if node in visited:
            continue
        components += 1
        frontier = [node]
        visited.add(node)
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
    return components


def summarize_graph_topology(graph: Any) -> Dict[str, int]:
    edge_index = graph.edge_index.long()
    edge_attr = collapse_classes(getattr(graph, "edge_attr", None))
    unique_edges = unique_undirected_edges(edge_index, edge_attr)
    num_nodes = infer_num_nodes(graph)
    return {
        "num_atoms": num_nodes,
        "num_bonds": len(unique_edges),
        "connected_components": count_connected_components(num_nodes, unique_edges),
    }


def canonicalize_largest_fragment(mol: Optional[rdchem.Mol]) -> Dict[str, Any]:
    if mol is None:
        return {
            "valid": False,
            "analysis_mol": None,
            "canonical_smiles": None,
            "num_components": None,
        }

    smiles = mol2smiles(mol)
    if smiles is None:
        return {
            "valid": False,
            "analysis_mol": None,
            "canonical_smiles": None,
            "num_components": None,
        }

    try:
        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        num_components = len(mol_frags)
        largest_mol = max(mol_frags, default=mol, key=lambda fragment: fragment.GetNumAtoms())
        largest_smiles = mol2smiles(largest_mol)
    except (Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException, ValueError, RuntimeError):
        return {
            "valid": False,
            "analysis_mol": None,
            "canonical_smiles": None,
            "num_components": None,
        }
    except (AttributeError, TypeError):
        return {
            "valid": False,
            "analysis_mol": None,
            "canonical_smiles": None,
            "num_components": None,
        }

    if largest_smiles in {None, ""}:
        return {
            "valid": False,
            "analysis_mol": None,
            "canonical_smiles": None,
            "num_components": num_components,
        }

    return {
        "valid": True,
        "analysis_mol": largest_mol,
        "canonical_smiles": largest_smiles,
        "num_components": num_components,
    }


def derive_mmff_atom_types(mol: rdchem.Mol) -> List[str]:
    try:
        props = AllChem.MMFFGetMoleculeProperties(Chem.AddHs(deepcopy(mol)))
    except (AttributeError, RuntimeError, ValueError):
        return []
    if props is None:
        return []
    return [str(props.GetMMFFAtomType(atom.GetIdx())) for atom in mol.GetAtoms()]


def derive_dataset_specific_atom_features(mol: rdchem.Mol) -> Dict[str, List[str]]:
    features = {
        NODE_IS_IN_RING_ATTR: [str(atom.IsInRing()) for atom in mol.GetAtoms()],
        NODE_IS_AROMATIC_ATTR: [str(atom.GetIsAromatic()) for atom in mol.GetAtoms()],
        NODE_HYBRIDIZATION_ATTR: [atom.GetHybridization().name for atom in mol.GetAtoms()],
    }
    mmff_types = derive_mmff_atom_types(mol)
    if mmff_types:
        features[NODE_MMFF_TYPE_ATTR] = mmff_types
    return features


def atom_environment_token(atom: Chem.rdchem.Atom) -> str:
    return "|".join(
        [
            atom.GetSymbol(),
            f"charge={atom.GetFormalCharge()}",
            f"degree={atom.GetDegree()}",
            f"hs={atom.GetTotalNumHs()}",
            f"ring={int(atom.IsInRing())}",
            f"aromatic={int(atom.GetIsAromatic())}",
            f"hyb={atom.GetHybridization().name}",
        ]
    )


def extract_substructure_counts(mol: rdchem.Mol) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name, pattern in SUBSTRUCTURE_PATTERNS.items():
        if pattern is None:
            counts[name] = 0
            continue
        counts[name] = len(mol.GetSubstructMatches(pattern, uniquify=True))
    return counts


def build_molecule_profile(mol: rdchem.Mol) -> Dict[str, Any]:
    ring_info = mol.GetRingInfo()
    atom_rings = list(ring_info.AtomRings())
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    aromatic_rings = sum(
        1 for ring in atom_rings if all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring)
    )

    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    formal_charges = [str(atom.GetFormalCharge()) for atom in mol.GetAtoms()]
    degrees = [str(atom.GetDegree()) for atom in mol.GetAtoms()]
    total_hydrogens = [str(atom.GetTotalNumHs()) for atom in mol.GetAtoms()]
    atom_environments = [atom_environment_token(atom) for atom in mol.GetAtoms()]
    bond_types = [bond.GetBondType().name.lower() for bond in mol.GetBonds()]
    ring_sizes = [str(len(ring)) for ring in atom_rings]
    dataset_specific_atom_features = derive_dataset_specific_atom_features(mol)
    substructure_counts = extract_substructure_counts(mol)

    num_atoms = max(mol.GetNumAtoms(), 1)
    return {
        "atom_symbols": atom_symbols,
        "formal_charges": formal_charges,
        "molecular_charge": [str(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))],
        "element_charge": [f"{symbol}|charge={charge}" for symbol, charge in zip(atom_symbols, formal_charges)],
        "mmff_charge": [f"{kind}|charge={charge}" for kind, charge in
                        zip(dataset_specific_atom_features.get(NODE_MMFF_TYPE_ATTR, []), formal_charges)],
        "element_atom_details": [f"{atom.GetSymbol()}|ring={int(atom.IsInRing())}|"
                                 f"aromatic={int(atom.GetIsAromatic())}|hyb={atom.GetHybridization().name}"
                                 for atom in mol.GetAtoms()],
        "degrees": degrees,
        "total_hydrogens": total_hydrogens,
        "atom_environments": atom_environments,
        "bond_types": bond_types,
        "ring_sizes": ring_sizes,
        "dataset_specific_atom_features": dataset_specific_atom_features,
        "substructure_counts": substructure_counts,
        "substructure_presence": {name: count > 0 for name, count in substructure_counts.items()},
        "num_atoms": mol.GetNumAtoms(),
        "num_bonds": mol.GetNumBonds(),
        "num_rings": len(atom_rings),
        "aromatic_atom_fraction": aromatic_atoms / num_atoms,
        "aromatic_ring_count": aromatic_rings,
    }


class VocabularyError(ValueError):
    """A graph does not match its source vocabulary; do not count this as invalid chemistry."""


class ChemistryDecoder:
    """Decode only elements, formal charges and bonds, ignoring predicted auxiliary states."""

    def __init__(self, stats_path: Path):
        self.stats_path = stats_path
        self.stats = json.loads(stats_path.read_text())

    def graph_to_molecule(self, graph):
        atoms = collapse_classes(graph.x).tolist()
        charges = getattr(graph, "node_charges", None)
        if charges is None:
            raise VocabularyError("Missing node_charges: formal-charge comparisons require explicit charges")
        charges = collapse_classes(charges).tolist()
        if len(atoms) != len(charges):
            raise VocabularyError("Atom and charge arrays are not aligned")
        bonds = unique_undirected_edges(graph.edge_index, collapse_classes(graph.edge_attr))
        if any(label < 0 or label >= len(self.stats["atom_types"]) for label in atoms) or any(
                charge < 0 or charge >= len(self.stats["charges"]) for charge in charges) or any(
                kind < 0 or kind >= len(self.stats["bond_types"]) for _, kind in bonds):
            raise VocabularyError(f"Graph labels exceed the vocabulary in {self.stats_path}")
        mol = Chem.RWMol()
        try:
            for label, charge in zip(atoms, charges):
                atom = Chem.Atom(self.stats["atom_types"][label].split("_")[0])
                atom.SetFormalCharge(int(self.stats["charges"][charge]))
                mol.AddAtom(atom)
            for (left, right), kind in bonds:
                mol.AddBond(left, right, getattr(rdchem.BondType, self.stats["bond_types"][kind]))
        except IndexError as error:
            raise VocabularyError(f"Graph labels exceed the vocabulary in {self.stats_path}") from error
        return mol.GetMol()


def load_reference_cache(root: Path, split: Optional[str]):
    path = root / (split or "") / "processed/data.pt"
    if not path.is_file() or not (root / "raw/stats.json").is_file():
        raise FileNotFoundError(f"Missing cached dataset/vocabulary: {path}, {root / 'raw/stats.json'}")
    data, slices, data_class = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    dataset = InMemoryDataset()
    dataset._data = data_class.from_dict(data) if isinstance(data, dict) else data
    dataset.slices = slices
    return dataset


CATEGORICAL_FIELDS = (
    "atom_symbols", "formal_charges", "molecular_charge", "element_charge", "mmff_charge",
    "element_atom_details", "degrees", "total_hydrogens", "atom_environments", "bond_types",
    "ring_sizes", "substructure_occurrences",
)


def aggregate_graphs(
    graphs: Iterable[Any],
    decoder: Any,
    *,
    desc: str,
    total: Optional[int] = None,
) -> Dict[str, Any]:
    counters = {name: Counter() for name in CATEGORICAL_FIELDS}
    counters["dataset_specific_atom_features"] = {}

    raw = {
        "graph_num_atoms": [],
        "graph_num_bonds": [],
        "graph_connected_components": [],
        "valid_num_atoms": [],
        "valid_num_bonds": [],
        "valid_num_rings": [],
        "valid_num_components": [],
        "valid_aromatic_ring_count": [],
        "valid_aromatic_atom_fraction": [],
    }

    valid_smiles: List[str] = []
    substructure_presence_counts = {name: 0 for name in SUBSTRUCTURE_PATTERNS}

    num_input_graphs = 0
    num_conversion_failures = 0
    num_valid_molecules = 0
    num_mmff_molecules = 0
    num_mmff_atoms = 0

    for graph in tqdm(graphs, desc=desc, total=total):
        num_input_graphs += 1

        topology = summarize_graph_topology(graph)
        raw["graph_num_atoms"].append(topology["num_atoms"])
        raw["graph_num_bonds"].append(topology["num_bonds"])
        raw["graph_connected_components"].append(topology["connected_components"])

        try:
            mol = decoder.graph_to_molecule(graph)
        except VocabularyError:
            raise
        except (AttributeError, RuntimeError, TypeError, ValueError):
            mol = None
            num_conversion_failures += 1

        canonicalized = canonicalize_largest_fragment(mol)
        if not canonicalized["valid"]:
            continue

        analysis_mol = canonicalized["analysis_mol"]
        profile = build_molecule_profile(analysis_mol)

        num_valid_molecules += 1
        valid_smiles.append(canonicalized["canonical_smiles"])

        for field in CATEGORICAL_FIELDS:
            if field != "substructure_occurrences":
                update_counter(counters[field], profile[field])
        mmff_atoms = profile["dataset_specific_atom_features"].get(NODE_MMFF_TYPE_ATTR, [])
        num_mmff_molecules += bool(mmff_atoms)
        num_mmff_atoms += len(mmff_atoms)
        update_nested_counters(counters["dataset_specific_atom_features"], profile["dataset_specific_atom_features"])

        for name, count in profile["substructure_counts"].items():
            counters["substructure_occurrences"][name] += int(count)
            if count > 0:
                substructure_presence_counts[name] += 1

        raw["valid_num_atoms"].append(profile["num_atoms"])
        raw["valid_num_bonds"].append(profile["num_bonds"])
        raw["valid_num_rings"].append(profile["num_rings"])
        raw["valid_num_components"].append(int(canonicalized["num_components"]))
        raw["valid_aromatic_ring_count"].append(profile["aromatic_ring_count"])
        raw["valid_aromatic_atom_fraction"].append(profile["aromatic_atom_fraction"])

    unique_smiles = sorted(set(valid_smiles))
    uniqueness_fraction = (len(unique_smiles) / len(valid_smiles)) if valid_smiles else None
    valid_fraction = (num_valid_molecules / num_input_graphs) if num_input_graphs else None
    presence_distribution = {
        name: (count / num_valid_molecules) if num_valid_molecules else None
        for name, count in substructure_presence_counts.items()
    }

    return {
        "num_input_graphs": num_input_graphs,
        "num_valid_molecules": num_valid_molecules,
        "num_conversion_failures": num_conversion_failures,
        "mmff_coverage": {"molecules": num_mmff_molecules, "atoms": num_mmff_atoms,
                          "failed_valid_molecules": num_valid_molecules - num_mmff_molecules,
                          "valid_molecule_fraction": num_mmff_molecules / num_valid_molecules
                          if num_valid_molecules else None},
        "valid_fraction": valid_fraction,
        "num_unique_valid_smiles": len(unique_smiles),
        "uniqueness_fraction": uniqueness_fraction,
        "counts": {name: dict(counter) for name, counter in counters.items()
                   if name != "dataset_specific_atom_features"},
        "dataset_specific_atom_feature_counts": {
            name: dict(counter) for name, counter in counters["dataset_specific_atom_features"].items()},
        "categorical_denominators": {
            **{name: sum(counters[name].values()) for name in CATEGORICAL_FIELDS},
            "dataset_specific_atom_features": {
                name: sum(counter.values()) for name, counter in counters["dataset_specific_atom_features"].items()}},
        "categorical": {
            **{name: normalize_counter(counters[name]) for name in CATEGORICAL_FIELDS},
            "dataset_specific_atom_features": normalize_nested_counters(counters["dataset_specific_atom_features"]),
        },
        "substructure_presence": presence_distribution,
        "scalar_histograms": {name: dict(Counter(map(str, values))) for name, values in raw.items()
                              if name != "valid_aromatic_atom_fraction"},
        "scalars": {
            "graph_num_atoms": scalar_summary(raw["graph_num_atoms"]),
            "graph_num_bonds": scalar_summary(raw["graph_num_bonds"]),
            "graph_connected_components": scalar_summary(raw["graph_connected_components"]),
            "valid_num_atoms": scalar_summary(raw["valid_num_atoms"]),
            "valid_num_bonds": scalar_summary(raw["valid_num_bonds"]),
            "valid_num_rings": scalar_summary(raw["valid_num_rings"]),
            "valid_num_components": scalar_summary(raw["valid_num_components"]),
            "valid_aromatic_ring_count": scalar_summary(raw["valid_aromatic_ring_count"]),
            "valid_aromatic_atom_fraction": scalar_summary(raw["valid_aromatic_atom_fraction"]),
        },
        "internals": {
            "counters": counters,
            "raw": raw,
            "valid_smiles": valid_smiles,
            "unique_smiles": unique_smiles,
        },
    }


def compare_aggregates(reference: Dict[str, Any], generated: Dict[str, Any]) -> Dict[str, Any]:
    reference_counters = reference["internals"]["counters"]
    generated_counters = generated["internals"]["counters"]
    reference_raw = reference["internals"]["raw"]
    generated_raw = generated["internals"]["raw"]

    reference_unique_smiles = set(reference["internals"]["unique_smiles"])
    generated_unique_smiles = generated["internals"]["unique_smiles"]
    novel_smiles = [smiles for smiles in generated_unique_smiles if smiles not in reference_unique_smiles]
    novelty_vs_reference = len(novel_smiles) / len(generated_unique_smiles) if generated_unique_smiles else None

    return {
        "sampling": {
            "valid_fraction_gap": None
            if reference["valid_fraction"] is None or generated["valid_fraction"] is None
            else generated["valid_fraction"] - reference["valid_fraction"],
            "uniqueness_gap": None
            if reference["uniqueness_fraction"] is None or generated["uniqueness_fraction"] is None
            else generated["uniqueness_fraction"] - reference["uniqueness_fraction"],
            "novelty_vs_reference": novelty_vs_reference,
            "num_novel_unique_smiles": len(novel_smiles),
        },
        "categorical_tvd": {
            **{name: total_variation_distance(reference_counters[name], generated_counters[name])
               for name in CATEGORICAL_FIELDS},
            "dataset_specific_atom_features": nested_tvd(
                reference_counters["dataset_specific_atom_features"],
                generated_counters["dataset_specific_atom_features"]),
        },
        "substructure_presence_gap": {
            name: None
            if reference["substructure_presence"].get(name) is None or generated["substructure_presence"].get(name) is None
            else generated["substructure_presence"][name] - reference["substructure_presence"][name]
            for name in sorted(set(reference["substructure_presence"]) | set(generated["substructure_presence"]))
        },
        "scalar_distances": {
            "graph_num_atoms_wasserstein": discrete_wasserstein(
                reference_raw["graph_num_atoms"], generated_raw["graph_num_atoms"]
            ),
            "graph_num_bonds_wasserstein": discrete_wasserstein(
                reference_raw["graph_num_bonds"], generated_raw["graph_num_bonds"]
            ),
            "graph_connected_components_wasserstein": discrete_wasserstein(
                reference_raw["graph_connected_components"], generated_raw["graph_connected_components"]
            ),
            "valid_num_atoms_wasserstein": discrete_wasserstein(
                reference_raw["valid_num_atoms"], generated_raw["valid_num_atoms"]
            ),
            "valid_num_bonds_wasserstein": discrete_wasserstein(
                reference_raw["valid_num_bonds"], generated_raw["valid_num_bonds"]
            ),
            "valid_num_rings_wasserstein": discrete_wasserstein(
                reference_raw["valid_num_rings"], generated_raw["valid_num_rings"]
            ),
            "valid_num_components_wasserstein": discrete_wasserstein(
                reference_raw["valid_num_components"], generated_raw["valid_num_components"]
            ),
            "valid_aromatic_ring_count_wasserstein": discrete_wasserstein(
                reference_raw["valid_aromatic_ring_count"], generated_raw["valid_aromatic_ring_count"]
            ),
            "valid_aromatic_atom_fraction_mean_gap": None
            if reference["scalars"]["valid_aromatic_atom_fraction"]["mean"] is None
            or generated["scalars"]["valid_aromatic_atom_fraction"]["mean"] is None
            else generated["scalars"]["valid_aromatic_atom_fraction"]["mean"]
            - reference["scalars"]["valid_aromatic_atom_fraction"]["mean"],
        },
    }


def emit_warning(message: str, messages: List[str]) -> None:
    messages.append(message)
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


def unwrap_config(value):
    return value.get("value") if isinstance(value, dict) and "value" in value else value


def read_run_metadata(run_dir: Path, checkpoints_root: Path) -> Dict[str, Any]:
    """Read seed from saved config/arguments, never from the vN directory counter."""
    candidates = [run_dir / "config.yaml", run_dir / ".hydra/config.yaml"]
    wandb_roots = {checkpoints_root / "wandb", run_dir.parent.parent / "wandb"}
    for root in sorted(wandb_roots):
        candidates.extend(sorted(root.glob(f"run-*-{run_dir.name}/files/config.yaml")))
    evidence = []
    dataset_names = set()
    for path in candidates:
        if not path.is_file():
            continue
        config = yaml.safe_load(path.read_text()) or {}
        value = unwrap_config(config.get("seed"))
        if value is not None:
            evidence.append((int(value), str(path)))
        task = unwrap_config(config.get("task", {})) or {}
        if task.get("dataset", {}).get("name"):
            dataset_names.add(task["dataset"]["name"])
    for root in sorted(wandb_roots):
        for path in sorted(root.glob(f"run-*-{run_dir.name}/files/wandb-metadata.json")):
            metadata = json.loads(path.read_text())
            for arg in metadata.get("args", []):
                match = re.fullmatch(r"\+{0,2}seed=(-?\d+)", arg)
                if match:
                    evidence.append((int(match.group(1)), str(path)))
    seeds = {seed for seed, _ in evidence}
    if len(seeds) > 1 or len(dataset_names) > 1:
        raise ValueError(f"Conflicting run metadata in {run_dir}: {evidence}, datasets={dataset_names}")
    return {"seed": next(iter(seeds), None), "seed_sources": sorted({path for _, path in evidence}),
            "dataset_name": next(iter(dataset_names), None)}


def generated_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted({file for pattern in GENERATED_FILE_PATTERNS for file in path.rglob(pattern)})


def discover_runs(paths: List[Path], checkpoints_root: Path, expected_seeds: List[int],
                  messages: List[str]) -> List[Dict[str, Any]]:
    by_run = {}
    for path in paths:
        if not path.exists():
            emit_warning(f"Generated path does not exist: {path}", messages)
            continue
        for file in generated_files(path):
            by_run.setdefault(file.parent.resolve(), set()).add(file.resolve())
    by_seed = {}
    for directory, files in sorted(by_run.items()):
        files = sorted(files, key=lambda file: (file.stat().st_mtime_ns, file.name), reverse=True)
        chosen = files[0]
        if len(files) > 1:
            emit_warning(f"Multiple generations in {directory}; using latest file {chosen.name}", messages)
        metadata = read_run_metadata(directory, checkpoints_root)
        seed = metadata["seed"]
        if seed is None:
            emit_warning(f"Skipping {directory}: training seed is unknown (saved config/W&B metadata required)", messages)
            continue
        if seed not in expected_seeds:
            emit_warning(f"Skipping {directory}: seed {seed} is outside expected seeds {expected_seeds}", messages)
            continue
        entry = {**metadata, "run_directory": str(directory), "generated_file": str(chosen)}
        by_seed.setdefault(seed, []).append(entry)
    result = []
    for seed, entries in sorted(by_seed.items()):
        entries.sort(key=lambda entry: (Path(entry["generated_file"]).stat().st_mtime_ns,
                                        entry["generated_file"]), reverse=True)
        if len(entries) > 1:
            emit_warning(f"Multiple runs for seed {seed}; using latest generation {entries[0]['generated_file']}", messages)
        result.append(entries[0])
    return result


def representation_name(dataset_name: str) -> str:
    return {"default": "base", "MMFF": "mmff", "atom_details": "atom_details"}[infer_atom_types_repr(dataset_name)]


def experiment_paths(args, family: str, messages: List[str]) -> Dict[str, List[Path]]:
    if args.generated:
        result = {}
        for argument in args.generated:
            name, separator, path = argument.partition("=")
            if not separator or name not in {"base", "mmff", "atom_details"}:
                name, path = representation_name(args.reference_dataset), argument
            result.setdefault(name, []).append(Path(path))
        return result
    result = {}
    for name in args.representations:
        stem = f"{family}-{args.model}"
        suffixes = {"base": [""], "mmff": ["-mmff-v2", "-mmff"],
                    "atom_details": ["-atom-details-v2", "-atom_details-v2", "-atom_details", "-atom-details"]}[name]
        candidates = [args.checkpoints_root / (stem + suffix) for suffix in suffixes]
        available = [path for path in candidates if path.exists() and generated_files(path)]
        if len(available) > 1:
            emit_warning(f"{name}: multiple experiment folders; using {available[0]}, ignoring {available[1:]}. "
                         "Use --generated NAME=PATH to choose explicitly.", messages)
        result[name] = [available[0]] if available else []
    return result


def numeric_summary(values: List[float]) -> Dict[str, Any]:
    values = [float(value) for value in values if value is not None]
    return {"mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None, "n": len(values)}


def summarize_tree(records: List[Dict], fill_missing_zero: bool = False) -> Dict:
    keys = sorted({key for record in records for key in record})
    result = {}
    for key in keys:
        values = [record[key] for record in records if key in record]
        if values and isinstance(values[0], dict):
            result[key] = summarize_tree([record.get(key, {}) for record in records], fill_missing_zero)
        else:
            result[key] = numeric_summary([
                record.get(key, 0 if fill_missing_zero else None) for record in records])
    return result


def flat_distributions(aggregate: Dict) -> Dict[str, Dict[str, float]]:
    result = {name: aggregate["categorical"][name] for name in CATEGORICAL_FIELDS}
    result.update(aggregate["categorical"]["dataset_specific_atom_features"])
    result["substructure_presence"] = aggregate["substructure_presence"]
    for name, histogram in aggregate.get("scalar_histograms", {}).items():
        result[name] = normalize_counter(Counter(histogram))
    return result


def summarize_runs(runs: List[Dict]) -> Dict:
    aggregates = [run["generated"] for run in runs]
    distributions = [flat_distributions(aggregate) for aggregate in aggregates]
    fields = sorted({field for record in distributions for field in record})
    return {
        "num_seeds": len(runs),
        "num_input_graphs": sum(aggregate["num_input_graphs"] for aggregate in aggregates),
        "num_valid_molecules": sum(aggregate["num_valid_molecules"] for aggregate in aggregates),
        "sampling": summarize_tree([{key: aggregate[key] for key in ["valid_fraction", "uniqueness_fraction"]}
                                    for aggregate in aggregates]),
        "comparison": summarize_tree([run["comparison"] for run in runs]),
        "distributions": {
            # Empty distributions (e.g. failed MMFF or no rings) are undefined, not all-zero probabilities.
            field: summarize_tree([record[field] for record in distributions
                                   if record.get(field) and any(v is not None for v in record[field].values())], True)
            for field in fields},
    }


def write_comparison_plots(report: Dict, output_dir: Path, max_categories: int) -> List[str]:
    reference = flat_distributions(report["reference"])
    groups = {name: [flat_distributions(run["generated"]) for run in entry["runs"]]
              for name, entry in report["representations"].items() if entry["runs"]}
    fields = sorted(set(reference) | {key for runs in groups.values() for run in runs for key in run})
    paths = []
    colors = {"base": "#4477AA", "mmff": "#EE7733", "atom_details": "#228833"}
    for field in fields:
        ref = reference.get(field, {})
        generated = {name: [record[field] for record in records if record.get(field)
                            and any(v is not None for v in record[field].values())]
                     for name, records in groups.items()}
        generated = {name: records for name, records in generated.items() if records}
        labels = {key for record in [ref] + [record for records in generated.values() for record in records]
                  for key, value in record.items() if value is not None}
        if not labels:
            continue
        scores = {key: (ref.get(key) or 0) + sum(statistics.mean(record.get(key) or 0 for record in records)
                                                  for records in generated.values()) for key in labels}
        selected = sorted(labels, key=lambda key: (-scores[key], key))[:max_categories]
        others = labels - set(selected)
        # Presence rates are independent Bernoulli rates, so summing omitted categories is meaningless.
        include_other = bool(others) and field != "substructure_presence"
        shown = selected + (["Other (omitted categories)"] if include_other else [])
        fig, ax = plt.subplots(figsize=(max(9, min(22, len(shown) * 0.75)), 5.5))
        width = 0.8 / (1 + len(generated))
        def values(record):
            return [(record.get(key) or 0) for key in selected] + (
                [sum(record.get(key) or 0 for key in others)] if include_other else [])
        offset = -(len(generated) / 2) * width
        if ref and any(value is not None for value in ref.values()):
            ax.bar([i + offset for i in range(len(shown))], values(ref), width,
                   label=f"Dataset ({report['metadata']['reference_split'] or 'full'})", color="#999999")
        for index, (name, records) in enumerate(generated.items(), start=1):
            vectors = [values(record) for record in records]
            stats = [numeric_summary(list(column)) for column in zip(*vectors)]
            ax.bar([i + offset + index * width for i in range(len(shown))],
                   [item["mean"] for item in stats], width,
                   yerr=[item["std"] or 0 for item in stats], capsize=2,
                   label=f"{name} (n={len(records)} seeds)", color=colors[name])
        ax.set_xticks(range(len(shown)))
        ax.set_xticklabels(shown, rotation=45, ha="right")
        ax.set_ylabel("Fraction of valid molecules" if field in {"substructure_presence", "molecular_charge"}
                      else "Relative frequency")
        ax.set_title(field.replace("_", " ") + " — seed mean ± sample standard deviation")
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"{sanitize_plot_filename(field)}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(str(path))
    return paths


def strip_internal_fields(aggregate: Dict) -> Dict:
    return {key: value for key, value in aggregate.items() if key != "internals"}


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    dataset_name = resolve_dataset_name(args.reference_dataset)
    family = re.sub(r"_(mmff|atom_details)$", "", dataset_name)
    split = normalize_split(args.reference_split)
    root = Path(resolve_dataset_root(dataset_name, args.reference_root))
    output = args.output or Path("outputs/distribution_comparison") / (
        f"{family}_{args.model}_{split or 'full'}_{datetime.now():%Y-%m-%d_%H-%M-%S}") / "report.json"
    messages = []
    paths = experiment_paths(args, family, messages)
    selected = {}
    for name, folders in paths.items():
        selected[name] = discover_runs(folders, args.checkpoints_root, args.expected_seeds, messages)
        found = [run["seed"] for run in selected[name]]
        missing = sorted(set(args.expected_seeds) - set(found))
        print(f"{name}: available seeds={found}; missing seeds={missing}", flush=True)
        if missing:
            emit_warning(f"{name}: missing seeds {missing}; analyzing only available seeds {found}", messages)
        for run in selected[name]:
            print(f"  seed={run['seed']}: {run['generated_file']}", flush=True)
    if not any(selected.values()):
        raise ValueError("No generated runs with identifiable training seeds were found")

    print(f"Loading reference {dataset_name}/{split or 'full'} from {root}", flush=True)
    reference_dataset = load_reference_cache(root, split)
    reference_size = len(reference_dataset)
    indices = select_indices(reference_size, args.max_reference, args.seed)
    reference = aggregate_graphs((reference_dataset[index] for index in indices),
                                 ChemistryDecoder(root / "raw/stats.json"), desc="Reference", total=len(indices))
    if reference["mmff_coverage"]["failed_valid_molecules"]:
        emit_warning(f"Reference: MMFF unavailable for {reference['mmff_coverage']['failed_valid_molecules']} "
                     "valid molecules", messages)
    if not reference["num_valid_molecules"]:
        raise ValueError("The reference contains no valid molecules; chemical comparison is undefined")
    del reference_dataset
    report = {
        "schema_version": 2,
        "metadata": {"reference_dataset": dataset_name, "reference_root": str(root), "reference_split": split,
                     "reference_total_graphs": reference_size,
                     "expected_seeds": args.expected_seeds, "model": args.model,
                     "max_reference": args.max_reference, "max_generated_per_seed": args.max_generated,
                     "subsampling_seed": args.seed, "output_file": str(output), "rdkit_version": Chem.rdBase.rdkitVersion},
        "notes": [
            "All descriptors (MMFF and atom_details included) are recomputed with RDKit for every representation.",
            "Atom frequencies count atoms, bond frequencies count bonds, ring sizes count rings; larger molecules contribute more atoms/bonds.",
            "Molecular charge and substructure presence count valid molecules. Occurrence frequencies count SMARTS matches (groups can overlap).",
            "Chemical profiles use valid largest sanitized fragments. Raw graph topology includes invalid and disconnected graphs.",
            "Repeated generated molecules contribute repeatedly to frequencies; uniqueness is reported separately.",
            "MMFF distributions use only molecules with available MMFF parameters; coverage and denominators are reported per run.",
            "Seed summaries give each available seed equal weight. std is the sample standard deviation (n-1), null for fewer than two seeds.",
            "TVD is in [0,1]; Wasserstein distances use the descriptor units. These are descriptive distances, not significance tests.",
            "novelty_vs_reference compares against the selected split, unlike training-log novelty against the training set.",
        ],
        "reference": strip_internal_fields(reference), "representations": {}, "warnings": messages,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for name, runs in selected.items():
        entry = {"available_seeds": [run["seed"] for run in runs],
                 "missing_seeds": sorted(set(args.expected_seeds) - {run["seed"] for run in runs}), "runs": []}
        report["representations"][name] = entry
        for run in runs:
            source_name = run["dataset_name"] or family + {"base": "", "mmff": "_mmff", "atom_details": "_atom_details"}[name]
            if representation_name(source_name) != name or re.sub(r"_(mmff|atom_details)$", "", resolve_dataset_name(source_name)) != family:
                raise ValueError(f"Run dataset {source_name} does not match requested {family}/{name}")
            source_root = root if source_name == dataset_name else Path(resolve_dataset_root(resolve_dataset_name(source_name), None))
            stats_path = source_root / "raw/stats.json"
            print(f"Loading {name}, seed {run['seed']}: {run['generated_file']}", flush=True)
            graphs = load_graph_collection(Path(run["generated_file"]))
            total_graphs = len(graphs)
            chosen = maybe_subsample(graphs, args.max_generated, args.seed)
            aggregate = aggregate_graphs(chosen, ChemistryDecoder(stats_path),
                                         desc=f"{name}/seed={run['seed']}", total=len(chosen))
            del graphs, chosen
            gc.collect()
            comparison = compare_aggregates(reference, aggregate)
            entry["runs"].append({**run, "source_vocabulary": str(stats_path), "total_saved_graphs": total_graphs,
                                  "generated": strip_internal_fields(aggregate), "comparison": comparison})
            if not aggregate["num_valid_molecules"]:
                emit_warning(f"{name}/seed={run['seed']}: no valid molecules; chemical distributions are undefined", messages)
            if aggregate["mmff_coverage"]["failed_valid_molecules"]:
                emit_warning(f"{name}/seed={run['seed']}: MMFF unavailable for "
                             f"{aggregate['mmff_coverage']['failed_valid_molecules']} valid molecules", messages)
            print(f"  analyzed={aggregate['num_input_graphs']}/{total_graphs}, "
                  f"valid={aggregate['num_valid_molecules']}, MMFF={aggregate['mmff_coverage']['molecules']}", flush=True)
            del aggregate
        entry["summary"] = summarize_runs(entry["runs"])
        output.write_text(json.dumps(round_floats(report), indent=2, allow_nan=False))
    report["metadata"]["plot_files"] = write_comparison_plots(report, output.parent, args.plot_top_k)
    output.write_text(json.dumps(round_floats(report), indent=2, allow_nan=False))
    print("\nDistances against the same reference (mean ± sample std over available seeds):")
    for name, entry in report["representations"].items():
        summary = entry["summary"]
        print(f"{name}: {summary['num_seeds']} seeds, {summary['num_input_graphs']} analyzed graphs")
        metrics = summary["comparison"].get("categorical_tvd", {})
        for field, stat in {**{key: value for key, value in metrics.items() if key != 'dataset_specific_atom_features'},
                            **metrics.get("dataset_specific_atom_features", {})}.items():
            print(f"  {field} TVD: {stat['mean']} ± {stat['std']} (n={stat['n']})")
    print(f"Report and plots: {output}")


if __name__ == "__main__":
    main()
