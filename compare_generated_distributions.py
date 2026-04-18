#!/usr/bin/env python3
"""Compare generated molecular graph distributions against a reference dataset.

This script is intentionally standalone: it does not change dataset preprocessing
or model code. It reuses the existing dataset registries and graph-to-molecule
decoder already present in the project.

Example:
    python compare_generated_distributions.py \
        --reference-dataset qm9_mmff \
        --generated checkpoints/qm9-d4-mmff/v0_32efk23c/ \
        --reference-split test
"""

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import warnings

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

import src.data.datasets as datasets
from src.data.utils.storing import load_file


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

LEGACY_AUXILIARY_NODE_ATTR = "node_atom_types_repr"


def parse_args() -> Any:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dataset",
        required=True,
        help="Registered dataset name, for example qm9, qm9_mmff, gdb13, zinc250k_mmff.",
    )
    parser.add_argument(
        "--reference-root",
        default=None,
        help="Override the reference dataset root. If omitted, the script tries project defaults.",
    )
    parser.add_argument(
        "--reference-split",
        default="test",
        help="Reference split to load. Use none/all/full to use the whole dataset. Default: test.",
    )
    parser.add_argument(
        "--generated",
        required=True,
        help="Generated graphs file, or a directory containing generated_graphs*.pkl / *.pt.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report path. If omitted, a timestamped report is written in the current directory.",
    )
    parser.add_argument(
        "--max-reference",
        type=int,
        default=None,
        help="Optional cap on the number of reference molecules processed after loading the dataset.",
    )
    parser.add_argument(
        "--max-generated",
        type=int,
        default=None,
        help="Optional cap on the number of generated molecules processed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used when subsampling molecules with --max-reference or --max-generated.",
    )
    return parser.parse_args()


def normalize_split(split: Optional[str]) -> Optional[str]:
    if split is None:
        return None
    split_lower = split.lower()
    if split_lower in {"none", "all", "full", "entire"}:
        return None
    return split


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

    candidates = DATASET_ROOT_CANDIDATES.get(dataset_name, [])
    if not candidates:
        candidates = [f"datasets/{dataset_name}"]

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    return candidates[0]


def resolve_generated_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_file():
        return path

    if not path.exists():
        raise FileNotFoundError(f"Generated path not found: {path}")

    matches: List[Path] = []
    for pattern in GENERATED_FILE_PATTERNS:
        matches.extend(path.rglob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No generated graph file found under {path}. Expected one of: {', '.join(GENERATED_FILE_PATTERNS)}"
        )

    matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[0]


def load_graph_collection(path: Path) -> List[Any]:
    graphs = load_file(str(path))
    if isinstance(graphs, list):
        return graphs
    if hasattr(graphs, "to_data_list"):
        return graphs.to_data_list()
    raise TypeError(f"Unsupported generated graph container type: {type(graphs)!r}")


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


def unique_undirected_edges(edge_index: torch.Tensor, edge_attr: torch.Tensor) -> List[Tuple[Tuple[int, int], int]]:
    seen: Dict[Tuple[int, int], int] = {}
    for pair, bond_type in zip(edge_index.t().tolist(), edge_attr.tolist()):
        src, dst = int(pair[0]), int(pair[1])
        key = (src, dst) if src <= dst else (dst, src)
        if key[0] == key[1]:
            continue
        if key not in seen:
            seen[key] = int(bond_type)
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


def decode_values(values: Optional[torch.Tensor], decoder: Optional[Dict[int, Any]]) -> List[str]:
    if values is None:
        return []
    if decoder is None:
        return [str(value) for value in values.tolist()]
    return [str(decoder[int(value)]) for value in values.tolist()]


def decode_auxiliary_node_states(graph: Any, decoder: Any) -> Tuple[List[str], Dict[str, List[str]]]:
    flattened_values: List[str] = []
    values_by_attr: Dict[str, List[str]] = {}

    for attr_name, attr_decoder in getattr(decoder, "auxiliary_node_state_decoders", {}).items():
        attr_values = collapse_classes(getattr(graph, attr_name, None))
        if attr_values is None:
            continue
        decoded_attr_values = decode_values(attr_values, attr_decoder)
        values_by_attr[attr_name] = decoded_attr_values
        flattened_values.extend(f"{attr_name}={value}" for value in decoded_attr_values)

    if flattened_values:
        return flattened_values, values_by_attr

    legacy_values = collapse_classes(getattr(graph, LEGACY_AUXILIARY_NODE_ATTR, None))
    if legacy_values is None:
        return [], {}

    legacy_decoder = getattr(decoder, "atom_types_repr_decoder", None)
    decoded_legacy_values = decode_values(legacy_values, legacy_decoder)
    return (
        [f"{LEGACY_AUXILIARY_NODE_ATTR}={value}" for value in decoded_legacy_values],
        {LEGACY_AUXILIARY_NODE_ATTR: decoded_legacy_values},
    )


def update_nested_counters(counters: Dict[str, Counter], values_by_attr: Dict[str, Iterable[str]]) -> None:
    for attr_name, values in values_by_attr.items():
        counter = counters.setdefault(attr_name, Counter())
        update_counter(counter, values)


def normalize_nested_counters(counters: Dict[str, Counter]) -> Dict[str, Dict[str, float]]:
    return {
        attr_name: normalize_counter(counter)
        for attr_name, counter in sorted(counters.items())
    }


def nested_tvd(
    counters_a: Dict[str, Counter],
    counters_b: Dict[str, Counter],
) -> Dict[str, Optional[float]]:
    metric_by_attr: Dict[str, Optional[float]] = {}
    for attr_name in sorted(set(counters_a) | set(counters_b)):
        metric_by_attr[attr_name] = total_variation_distance(
            counters_a.get(attr_name, Counter()),
            counters_b.get(attr_name, Counter()),
        )
    return metric_by_attr


def sanitize_plot_filename(name: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in name)


def plot_categorical_distribution(
    reference_distribution: Dict[str, float],
    generated_distribution: Dict[str, float],
    title: str,
    output_path: Path,
) -> None:
    labels = sorted(set(reference_distribution) | set(generated_distribution))
    reference_values = [reference_distribution.get(label, 0.0) for label in labels]
    generated_values = [generated_distribution.get(label, 0.0) for label in labels]

    if not labels:
        return

    figure_width = max(8.0, min(18.0, 1.2 * len(labels)))
    fig, ax = plt.subplots(figsize=(figure_width, 4.8))
    positions = list(range(len(labels)))
    bar_width = 0.4
    ax.bar(
        [position - bar_width / 2 for position in positions],
        reference_values,
        width=bar_width,
        color="#7f8c8d",
        label="Reference",
    )
    ax.bar(
        [position + bar_width / 2 for position in positions],
        generated_values,
        width=bar_width,
        color="#2c7fb8",
        label="Generated",
    )
    ax.set_title(title)
    ax.set_ylabel("Probability")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.0, max(reference_values + generated_values) * 1.1 if labels else 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_distribution_plots(report: Dict[str, Any], output_path: Path) -> List[str]:
    plot_dir = output_path.with_suffix("")
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_paths: List[str] = []

    reference_categorical = report["reference"]["categorical"]
    generated_categorical = report["generated"]["categorical"]
    atom_types_plot = plot_dir / "atom_types_reference_vs_generated.png"
    plot_categorical_distribution(
        reference_categorical["atom_types"],
        generated_categorical["atom_types"],
        title="Atom type distribution: reference vs generated",
        output_path=atom_types_plot,
    )
    plot_paths.append(str(atom_types_plot))

    for attr_name, distribution in generated_categorical.get("auxiliary_node_states", {}).items():
        plot_path = plot_dir / f"{sanitize_plot_filename(attr_name)}_reference_vs_generated.png"
        plot_categorical_distribution(
            reference_categorical.get("auxiliary_node_states", {}).get(attr_name, {}),
            distribution,
            title=f"{attr_name} distribution: reference vs generated",
            output_path=plot_path,
        )
        plot_paths.append(str(plot_path))

    return plot_paths


def build_molecule_summary(mol: Optional[Any]) -> Dict[str, Any]:
    if mol is None:
        return {
            "valid": False,
            "num_rings": None,
            "ring_sizes": [],
            "aromatic_atom_fraction": None,
            "aromatic_ring_count": None,
        }

    ring_info = mol.GetRingInfo()
    atom_rings = list(ring_info.AtomRings())
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    aromatic_rings = 0
    for ring in atom_rings:
        if all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring):
            aromatic_rings += 1

    num_atoms = max(mol.GetNumAtoms(), 1)
    return {
        "valid": True,
        "num_rings": len(atom_rings),
        "ring_sizes": [len(ring) for ring in atom_rings],
        "aromatic_atom_fraction": aromatic_atoms / num_atoms,
        "aromatic_ring_count": aromatic_rings,
    }


def summarize_graph(graph: Any, decoder: Any) -> Dict[str, Any]:
    atom_indices = collapse_classes(graph.x)
    edge_indices = graph.edge_index.long()
    edge_attr = collapse_classes(graph.edge_attr)
    charge_indices = collapse_classes(getattr(graph, "node_charges", None))

    unique_edges = unique_undirected_edges(edge_indices, edge_attr)

    atom_types = decode_values(atom_indices, getattr(decoder, "atom_decoder", None))
    aux_atom_types, auxiliary_node_states = decode_auxiliary_node_states(graph, decoder)
    charge_values = decode_values(charge_indices, getattr(decoder, "charge_decoder", None))

    bond_indices = torch.tensor([bond_type for _, bond_type in unique_edges], dtype=torch.long)
    bond_types = decode_values(bond_indices, getattr(decoder, "bond_decoder", None))

    mol = None
    try:
        mol = decoder.graph_to_molecule(graph)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        mol = None

    mol_summary = build_molecule_summary(mol)

    return {
        "num_atoms": int(graph.num_nodes),
        "num_bonds": len(unique_edges),
        "connected_components": count_connected_components(graph.num_nodes, unique_edges),
        "atom_types": atom_types,
        "aux_atom_types": aux_atom_types,
        "auxiliary_node_states": auxiliary_node_states,
        "bond_types": bond_types,
        "charges": charge_values,
        **mol_summary,
    }


def update_counter(counter: Counter, values: Iterable[str]) -> None:
    for value in values:
        counter[str(value)] += 1


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


def normalize_counter(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in sorted(counter.items())}


def total_variation_distance(counter_a: Counter, counter_b: Counter) -> Optional[float]:
    if sum(counter_a.values()) == 0 or sum(counter_b.values()) == 0:
        return None

    dist_a = normalize_counter(counter_a)
    dist_b = normalize_counter(counter_b)
    keys = set(dist_a) | set(dist_b)
    return 0.5 * sum(abs(dist_a.get(key, 0.0) - dist_b.get(key, 0.0)) for key in keys)


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


def round_floats(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: round_floats(item, digits=digits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item, digits=digits) for item in value]
    return value


def aggregate_summaries(summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    atom_types = Counter()
    aux_atom_types = Counter()
    auxiliary_node_states: Dict[str, Counter] = {}
    bond_types = Counter()
    charges = Counter()
    ring_sizes = Counter()

    num_atoms: List[int] = []
    num_bonds: List[int] = []
    num_rings: List[int] = []
    connected_components: List[int] = []
    aromatic_atom_fraction: List[float] = []
    aromatic_ring_count: List[int] = []
    valid_molecules = 0

    for summary in summaries:
        update_counter(atom_types, summary["atom_types"])
        update_counter(aux_atom_types, summary["aux_atom_types"])
        update_nested_counters(auxiliary_node_states, summary["auxiliary_node_states"])
        update_counter(bond_types, summary["bond_types"])
        update_counter(charges, summary["charges"])
        update_counter(ring_sizes, [str(size) for size in summary["ring_sizes"]])

        num_atoms.append(summary["num_atoms"])
        num_bonds.append(summary["num_bonds"])
        connected_components.append(summary["connected_components"])

        if summary["valid"]:
            valid_molecules += 1
            num_rings.append(int(summary["num_rings"]))
            aromatic_ring_count.append(int(summary["aromatic_ring_count"]))
            aromatic_atom_fraction.append(float(summary["aromatic_atom_fraction"]))

    return {
        "num_molecules": len(summaries),
        "valid_molecules": valid_molecules,
        "valid_fraction": (valid_molecules / len(summaries)) if summaries else None,
        "categorical": {
            "atom_types": normalize_counter(atom_types),
            "aux_atom_types": normalize_counter(aux_atom_types),
            "auxiliary_node_states": normalize_nested_counters(auxiliary_node_states),
            "bond_types": normalize_counter(bond_types),
            "charges": normalize_counter(charges),
            "ring_sizes": normalize_counter(ring_sizes),
        },
        "counters": {
            "atom_types": atom_types,
            "aux_atom_types": aux_atom_types,
            "auxiliary_node_states": auxiliary_node_states,
            "bond_types": bond_types,
            "charges": charges,
            "ring_sizes": ring_sizes,
        },
        "scalars": {
            "num_atoms": scalar_summary(num_atoms),
            "num_bonds": scalar_summary(num_bonds),
            "num_rings": scalar_summary(num_rings),
            "connected_components": scalar_summary(connected_components),
            "aromatic_atom_fraction": scalar_summary(aromatic_atom_fraction),
            "aromatic_ring_count": scalar_summary(aromatic_ring_count),
        },
        "raw": {
            "num_atoms": num_atoms,
            "num_bonds": num_bonds,
            "num_rings": num_rings,
            "connected_components": connected_components,
            "aromatic_atom_fraction": aromatic_atom_fraction,
            "aromatic_ring_count": aromatic_ring_count,
        },
    }


def aggregate_graphs(
    graphs: Iterable[Any],
    decoder: Any,
    *,
    desc: str,
    total: Optional[int] = None,
) -> Dict[str, Any]:
    atom_types = Counter()
    aux_atom_types = Counter()
    auxiliary_node_states: Dict[str, Counter] = {}
    bond_types = Counter()
    charges = Counter()
    ring_sizes = Counter()

    num_atoms: List[int] = []
    num_bonds: List[int] = []
    num_rings: List[int] = []
    connected_components: List[int] = []
    aromatic_atom_fraction: List[float] = []
    aromatic_ring_count: List[int] = []
    valid_molecules = 0
    num_molecules = 0

    for graph in tqdm(graphs, desc=desc, total=total):
        summary = summarize_graph(graph, decoder)
        num_molecules += 1

        update_counter(atom_types, summary["atom_types"])
        update_counter(aux_atom_types, summary["aux_atom_types"])
        update_nested_counters(auxiliary_node_states, summary["auxiliary_node_states"])
        update_counter(bond_types, summary["bond_types"])
        update_counter(charges, summary["charges"])
        update_counter(ring_sizes, [str(size) for size in summary["ring_sizes"]])

        num_atoms.append(summary["num_atoms"])
        num_bonds.append(summary["num_bonds"])
        connected_components.append(summary["connected_components"])

        if summary["valid"]:
            valid_molecules += 1
            num_rings.append(int(summary["num_rings"]))
            aromatic_ring_count.append(int(summary["aromatic_ring_count"]))
            aromatic_atom_fraction.append(float(summary["aromatic_atom_fraction"]))

    return {
        "num_molecules": num_molecules,
        "valid_molecules": valid_molecules,
        "valid_fraction": (valid_molecules / num_molecules) if num_molecules else None,
        "categorical": {
            "atom_types": normalize_counter(atom_types),
            "aux_atom_types": normalize_counter(aux_atom_types),
            "auxiliary_node_states": normalize_nested_counters(auxiliary_node_states),
            "bond_types": normalize_counter(bond_types),
            "charges": normalize_counter(charges),
            "ring_sizes": normalize_counter(ring_sizes),
        },
        "counters": {
            "atom_types": atom_types,
            "aux_atom_types": aux_atom_types,
            "auxiliary_node_states": auxiliary_node_states,
            "bond_types": bond_types,
            "charges": charges,
            "ring_sizes": ring_sizes,
        },
        "scalars": {
            "num_atoms": scalar_summary(num_atoms),
            "num_bonds": scalar_summary(num_bonds),
            "num_rings": scalar_summary(num_rings),
            "connected_components": scalar_summary(connected_components),
            "aromatic_atom_fraction": scalar_summary(aromatic_atom_fraction),
            "aromatic_ring_count": scalar_summary(aromatic_ring_count),
        },
        "raw": {
            "num_atoms": num_atoms,
            "num_bonds": num_bonds,
            "num_rings": num_rings,
            "connected_components": connected_components,
            "aromatic_atom_fraction": aromatic_atom_fraction,
            "aromatic_ring_count": aromatic_ring_count,
        },
    }


def compare_aggregates(reference: Dict[str, Any], generated: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "valid_fraction_gap": None
        if reference["valid_fraction"] is None or generated["valid_fraction"] is None
        else generated["valid_fraction"] - reference["valid_fraction"],
        "categorical": {
            "atom_types_tvd": total_variation_distance(
                reference["counters"]["atom_types"], generated["counters"]["atom_types"]
            ),
            "aux_atom_types_tvd": total_variation_distance(
                reference["counters"]["aux_atom_types"], generated["counters"]["aux_atom_types"]
            ),
            "auxiliary_node_states_tvd": nested_tvd(
                reference["counters"]["auxiliary_node_states"],
                generated["counters"]["auxiliary_node_states"],
            ),
            "bond_types_tvd": total_variation_distance(
                reference["counters"]["bond_types"], generated["counters"]["bond_types"]
            ),
            "charges_tvd": total_variation_distance(
                reference["counters"]["charges"], generated["counters"]["charges"]
            ),
            "ring_sizes_tvd": total_variation_distance(
                reference["counters"]["ring_sizes"], generated["counters"]["ring_sizes"]
            ),
        },
        "scalar": {
            "num_atoms_wasserstein": discrete_wasserstein(
                reference["raw"]["num_atoms"], generated["raw"]["num_atoms"]
            ),
            "num_bonds_wasserstein": discrete_wasserstein(
                reference["raw"]["num_bonds"], generated["raw"]["num_bonds"]
            ),
            "num_rings_wasserstein": discrete_wasserstein(
                reference["raw"]["num_rings"], generated["raw"]["num_rings"]
            ),
            "connected_components_wasserstein": discrete_wasserstein(
                reference["raw"]["connected_components"], generated["raw"]["connected_components"]
            ),
            "aromatic_ring_count_wasserstein": discrete_wasserstein(
                reference["raw"]["aromatic_ring_count"], generated["raw"]["aromatic_ring_count"]
            ),
            "aromatic_atom_fraction_mean_gap": None
            if reference["scalars"]["aromatic_atom_fraction"]["mean"] is None
            or generated["scalars"]["aromatic_atom_fraction"]["mean"] is None
            else generated["scalars"]["aromatic_atom_fraction"]["mean"]
            - reference["scalars"]["aromatic_atom_fraction"]["mean"],
        },
    }


def instantiate_reference_resources(
    dataset_name: str,
    dataset_root: str,
    include_charges: bool,
) -> Any:
    params = {
        "root": dataset_root,
        "random_splits": None,
        "include_pos": False,
        "include_charges": include_charges,
        "remove_hydrogens": True,
        "hard_remove_hydrogens": False,
        "kekulize": True,
        "atom_types_repr": infer_atom_types_repr(dataset_name),
    }
    return datasets.reg_dataresources.get_instance(dataset_name, params=params)


def make_output_path(output: Optional[str], dataset_name: str, reference_root: str) -> Path:
    if output is not None:
        return Path(output)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(reference_root) / f"distribution_comparison_{dataset_name}_{timestamp}.json"


def print_summary(report: Dict[str, Any]) -> None:
    metadata = report["metadata"]
    comparison = report["comparison"]
    reference = report["reference"]
    generated = report["generated"]

    print(f"Reference dataset: {metadata['reference_dataset']} (split={metadata['reference_split']})")
    print(f"Reference root:    {metadata['reference_root']}")
    print(f"Generated file:    {metadata['generated_file']}")
    print(f"Report path:       {metadata['output_file']}")
    print("")
    print(
        "Molecules:        "
        f"reference={reference['num_molecules']}, generated={generated['num_molecules']}"
    )
    print(
        "Valid fraction:   "
        f"reference={reference['valid_fraction']}, generated={generated['valid_fraction']}"
    )
    print("")
    print("Distances:")
    print(f"  atom types TVD:              {comparison['categorical']['atom_types_tvd']}")
    print(f"  auxiliary atom types TVD:    {comparison['categorical']['aux_atom_types_tvd']}")
    for attr_name, tvd in comparison["categorical"].get("auxiliary_node_states_tvd", {}).items():
        print(f"  {attr_name} TVD:              {tvd}")
    print(f"  bond types TVD:              {comparison['categorical']['bond_types_tvd']}")
    print(f"  ring sizes TVD:              {comparison['categorical']['ring_sizes_tvd']}")
    print(f"  num atoms Wasserstein:       {comparison['scalar']['num_atoms_wasserstein']}")
    print(f"  num bonds Wasserstein:       {comparison['scalar']['num_bonds_wasserstein']}")
    print(f"  num rings Wasserstein:       {comparison['scalar']['num_rings_wasserstein']}")
    print(f"  connected comps Wasserstein: {comparison['scalar']['connected_components_wasserstein']}")
    print("")


def strip_internal_fields(aggregate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in aggregate.items()
        if key not in {"counters", "raw"}
    }


def main() -> None:
    args = parse_args()

    dataset_name = resolve_dataset_name(args.reference_dataset)
    print(f"[1/6] Resolving generated graphs from: {args.generated}")

    generated_path = resolve_generated_path(args.generated)
    print(f"      Using generated file: {generated_path}")

    print("[2/6] Loading generated graphs")
    generated_graphs = load_graph_collection(generated_path)
    generated_graphs = maybe_subsample(generated_graphs, args.max_generated, args.seed)
    generated_has_charges = bool(generated_graphs) and hasattr(generated_graphs[0], "node_charges")
    print(f"      Loaded {len(generated_graphs)} generated graphs")

    reference_root = resolve_dataset_root(dataset_name, args.reference_root)
    reference_split = normalize_split(args.reference_split)
    print(f"[3/6] Preparing reference resources for dataset={dataset_name}, split={reference_split}")
    data_resources = instantiate_reference_resources(
        dataset_name=dataset_name,
        dataset_root=reference_root,
        include_charges=generated_has_charges,
    )

    print("[4/6] Loading reference dataset")
    reference_dataset = data_resources.get("dataset", split=reference_split)
    reference_indices = select_indices(len(reference_dataset), args.max_reference, args.seed)
    print(f"      Selected {len(reference_indices)} reference molecules out of {len(reference_dataset)}")

    decoder = data_resources.get("decoder")

    print("[5/6] Aggregating distributions")
    reference_aggregate = aggregate_graphs(
        (reference_dataset[index] for index in reference_indices),
        decoder,
        desc="Reference molecules",
        total=len(reference_indices),
    )
    generated_aggregate = aggregate_graphs(
        generated_graphs,
        decoder,
        desc="Generated molecules",
        total=len(generated_graphs),
    )
    comparison = compare_aggregates(reference_aggregate, generated_aggregate)

    output_path = make_output_path(args.output, dataset_name, reference_root)
    report = {
        "metadata": {
            "reference_dataset": dataset_name,
            "reference_root": reference_root,
            "reference_split": reference_split,
            "generated_file": str(generated_path),
            "generated_has_charges": generated_has_charges,
            "max_reference": args.max_reference,
            "max_generated": args.max_generated,
            "seed": args.seed,
            "output_file": str(output_path),
        },
        "reference": strip_internal_fields(reference_aggregate),
        "generated": strip_internal_fields(generated_aggregate),
        "comparison": comparison,
    }

    print(f"[6/6] Writing report to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_paths = write_distribution_plots(report, output_path)
    report["metadata"]["plot_files"] = plot_paths
    output_path.write_text(json.dumps(round_floats(report), indent=2), encoding="utf-8")
    print_summary(round_floats(report))


if __name__ == "__main__":
    main()