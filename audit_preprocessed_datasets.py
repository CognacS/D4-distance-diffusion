#!/usr/bin/env python3
"""Audit saved molecular datasets without invoking preprocessing or downloads.

Run with the project's d4-gen Python environment. Reports are JSON; input caches
are read only. Chemistry checks compare every QM9 molecule with its raw RDKit
source. Other datasets receive complete tensor checks and sampled chemistry checks.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import random
import shutil
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import ChemicalForceFields
import torch

# Register project data classes before deserializing the project's own caches.
import src.data.datasets  # noqa: F401


def read_json(path):
    return json.loads(Path(path).read_text())


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(path):
    data, slices, data_class = torch.load(
        str(path), map_location='cpu', weights_only=False, mmap=True,
    )
    if not isinstance(data, dict):
        data = data.to_dict()
    if slices is None:
        # PyG saves a single graph without slices or a batch dimension on y.
        data = dict(data)
        if 'y' in data:
            data['y'] = data['y'].unsqueeze(0)
        slices = {
            key: torch.tensor([0, value.shape[1 if key == 'edge_index' else 0]])
            for key, value in data.items()
        }
    return data, slices, data_class


def classes(value):
    return value.argmax(-1) if value.ndim == 2 else value.long()


def categorical_dims(stats):
    return {
        'x': stats['num_cls_nodes'],
        'edge_attr': stats['num_cls_edges'],
        'node_charges': stats['num_cls_charges'],
        **{key: len(values) for key, values in stats.get('auxiliary_node_state_values', {}).items()},
    }


def calculate_stats(data, slices, template):
    """Recompute statistics with the same no-bond convention as training."""
    stats = dict(template)
    nodes = slices['x'].diff()
    edges = slices['edge_index'].diff()
    stats['num_graphs'] = len(nodes)
    for name, lengths in [('nodes', nodes), ('edges', edges)]:
        stats[f'num_{name}_min'] = int(lengths.min())
        stats[f'num_{name}_max'] = int(lengths.max())
        stats[f'num_{name}_hist'] = dict(Counter(map(int, lengths)))
    marginals = {}
    for key, count in categorical_dims(template).items():
        values = data[key]
        histogram = (
            values.sum(0, dtype=torch.float64) if values.ndim == 2
            else torch.bincount(values.long(), minlength=count).double()
        )
        if key == 'edge_attr':
            missing = (nodes * (nodes - 1) - edges).sum()
            histogram = torch.cat([missing.reshape(1).double(), histogram])
        marginals[key] = (histogram / histogram.sum()).tolist()
    stats['marginals'] = marginals
    return stats


def audit_cache(path):
    data, slices, _ = load_cache(path)
    stats = read_json(path.with_name('stats.json'))
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    n_graphs = len(slices['x']) - 1
    nodes, edges = slices['x'].diff(), slices['edge_index'].diff()
    require(n_graphs == stats['num_graphs'], 'Graph count differs from stats.json')
    require(bool((nodes > 0).all()), 'Empty graphs')
    dims = categorical_dims(stats)
    for key, value in data.items():
        require(key in slices, f'{key}: missing slices')
        boundaries = slices[key]
        dim = 1 if key == 'edge_index' else 0
        require(len(boundaries) == n_graphs + 1, f'{key}: inconsistent graph count')
        require(int(boundaries[0]) == 0 and int(boundaries[-1]) == value.shape[dim], f'{key}: invalid slice bounds')
        require(bool((boundaries.diff() >= 0).all()), f'{key}: decreasing slices')
        require(bool(torch.isfinite(value).all()), f'{key}: nonfinite entries')
        if key.startswith('node_') or key == 'x':
            require(torch.equal(boundaries, slices['x']), f'{key}: node alignment differs')
        if key in dims:
            if value.ndim == 2:
                require(value.shape[1] == dims[key], f'{key}: wrong category dimension')
                require(bool(((value == 0) | (value == 1)).all() and (value.sum(-1) == 1).all()), f'{key}: invalid one-hot entries')
            else:
                require(value.ndim == 1 and bool(((value >= 0) & (value < dims[key])).all()), f'{key}: invalid category indices')
    require(data['node_pos'].shape[1:] == (3,), 'Positions must have three coordinates')
    require(data['y'].shape == (n_graphs, stats['num_cls_properties']), 'Wrong target dimensions')

    max_center_error = 0.0
    edge_labels = classes(data['edge_attr'])
    # Limit temporary indexing arrays while inspecting all graphs and edges.
    for start in range(0, n_graphs, 10000):
        end = min(n_graphs, start + 10000)
        node_counts, edge_counts = nodes[start:end], edges[start:end]
        batch = torch.arange(end - start).repeat_interleave(node_counts)
        pos = data['node_pos'][slices['x'][start]:slices['x'][end]]
        centers = torch.zeros(end - start, 3).index_add_(0, batch, pos)
        centers /= node_counts[:, None]
        max_center_error = max(max_center_error, float(centers.abs().max()))
        edge_batch = torch.arange(end - start).repeat_interleave(edge_counts)
        first, last = int(slices['edge_index'][start]), int(slices['edge_index'][end])
        edge_index = data['edge_index'][:, first:last]
        require(bool(((edge_index >= 0) & (edge_index < node_counts[edge_batch])).all()), f'Invalid edge endpoint in graphs {start}:{end}')
        require(bool((edge_index[0] != edge_index[1]).all()), f'Self loops in graphs {start}:{end}')
        width = int(node_counts.max())
        codes = edge_batch * width * width + edge_index[0] * width + edge_index[1]
        reversed_codes = edge_batch * width * width + edge_index[1] * width + edge_index[0]
        order, reverse_order = codes.argsort(), reversed_codes.argsort()
        require(torch.equal(codes[order], reversed_codes[reverse_order]), f'Asymmetric edges in graphs {start}:{end}')
        require(torch.equal(edge_labels[first:last][order], edge_labels[first:last][reverse_order]), f'Asymmetric bond labels in graphs {start}:{end}')
        require(len(codes.unique()) == len(codes), f'Duplicate edges in graphs {start}:{end}')
    require(max_center_error < 1e-4, 'Positions are not centered')

    actual = calculate_stats(data, slices, stats)
    for key, expected in actual.items():
        if key == 'marginals':
            for attr, values in expected.items():
                stored = stats.get(key, {}).get(attr, [])
                require(len(stored) == len(values) and np.allclose(stored, values, atol=1e-7, rtol=1e-6), f'{attr}: stale marginals')
        elif key.endswith('_hist'):
            require({str(k): v for k, v in expected.items()} == stats[key], f'{key}: stale histogram')
        else:
            require(expected == stats[key], f'{key}: stale statistic')
    return {'graphs': n_graphs, 'nodes': int(nodes.sum()), 'directed_edges': int(edges.sum()),
            'max_center_error': max_center_error, 'errors': errors}


def expected_qm9_splits(size, seed=0):
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    return {'valid': indices[:10000], 'test': indices[10000:20000], 'train': indices[20000:]}


def subset_cache(data, slices, indices):
    """Select complete graphs from a PyG cache, preserving local edge indices."""
    indices = torch.tensor(indices, dtype=torch.long)
    out, out_slices = {}, {}
    for key, value in data.items():
        lengths = slices[key].diff()[indices]
        boundaries = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
        offsets = (slices[key][indices] - boundaries[:-1]).repeat_interleave(lengths)
        gather = torch.arange(int(boundaries[-1])) + offsets
        out[key] = value.index_select(1 if key == 'edge_index' else 0, gather)
        out_slices[key] = boundaries
    return out, out_slices


def audit_splits(root):
    full_smiles = read_json(root / 'raw/smiles.json')
    report = {'split_sizes': {}, 'errors': [], 'warnings': []}
    split_smiles = {}
    for split in ['train', 'valid', 'val', 'test']:
        path = root / split / 'raw/smiles.json'
        if path.exists():
            split_smiles[split] = read_json(path)
            report['split_sizes'][split] = len(split_smiles[split])
    if root.name.startswith('qm9'):
        expected = expected_qm9_splits(len(full_smiles))
    elif root.name == 'gdb13':
        indices = list(range(len(full_smiles)))
        random.Random(0).shuffle(indices)
        size = int(len(indices) * 0.1)
        expected = {'valid': indices[:size], 'test': indices[size:2 * size], 'train': indices[2 * size:]}
    else:
        expected = None
    full_data, full_slices, _ = load_cache(root / 'processed/data.pt')
    if root.name == 'zinc':
        # ZINC keeps its predefined test flag in the first target column.
        test_mask = full_data['y'][:, 0].bool()
        test_indices = test_mask.nonzero(as_tuple=True)[0].tolist()
        others = (~test_mask).nonzero(as_tuple=True)[0].tolist()
        random.Random(0).shuffle(others)
        validation_size = int(len(others) * 0.16)
        expected = {'valid': others[:validation_size], 'train': others[validation_size:], 'test': test_indices}
    if expected is not None:
        for split, indices in expected.items():
            expected_smiles = [full_smiles[i] for i in indices]
            if split_smiles.get(split) != expected_smiles:
                report['errors'].append(f'{split}: SMILES do not match seed=0 and configured split sizes')
            path = root / split / 'processed/data.pt'
            if not path.exists():
                report['errors'].append(f'{split}: missing graph cache')
                continue
            saved, saved_slices, _ = load_cache(path)
            if len(saved_slices['x']) != len(indices) + 1:
                report['errors'].append(f'{split}: wrong graph count')
                continue
            if set(saved) != set(full_data):
                report['errors'].append(f'{split}: attributes differ from the full dataset')
            mismatches = set()
            for start in range(0, len(indices), 10000):
                end = min(start + 10000, len(indices))
                selected, selected_slices = subset_cache(full_data, full_slices, indices[start:end])
                for key in selected.keys() & saved.keys():
                    bounds = saved_slices[key][start:end + 1]
                    actual = saved[key].narrow(1 if key == 'edge_index' else 0, int(bounds[0]), int(bounds[-1] - bounds[0]))
                    if not torch.equal(bounds - bounds[0], selected_slices[key]) or not torch.equal(actual, selected[key]):
                        mismatches.add(key)
            report['errors'].extend(f'{split}/{key}: differs from the corresponding full dataset records' for key in sorted(mismatches))
    else:
        combined = Counter(s for name in ['train', 'valid', 'test'] for s in split_smiles.get(name, []))
        if combined != Counter(full_smiles):
            report['errors'].append('Split SMILES do not partition the full dataset')
    for left, right in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
        overlap = len(set(split_smiles.get(left, [])) & set(split_smiles.get(right, [])))
        report[f'{left}_{right}_shared_smiles'] = overlap
        if overlap:
            report['warnings'].append(f'{left}/{right}: {overlap} shared SMILES (source duplicates or overlapping splits; inspect record identities)')
    return report


def repair_qm9_splits(root):
    """Stage a seed-zero partition, validate it, and archive the old partition."""
    if not root.name.startswith('qm9'):
        raise ValueError('This repair only supports the configured QM9 partition')
    full_path = root / 'processed/data.pt'
    full_audit = audit_cache(full_path)
    if full_audit['errors']:
        raise ValueError(f'Cannot repair splits from an invalid full cache: {full_audit}')
    data, slices, data_class = load_cache(full_path)
    template = read_json(full_path.with_name('stats.json'))
    smiles = read_json(root / 'raw/smiles.json')
    if len(smiles) != full_audit['graphs'] or len(smiles) < 20000:
        raise ValueError('The full QM9 graph and SMILES caches must agree and contain at least 20000 records')
    split_indices = expected_qm9_splits(len(smiles))
    run_id = f'audit_{time.time_ns()}'
    staging = root / 'split_staging' / run_id
    backup = root / 'split_backups' / run_id
    for split, indices in split_indices.items():
        selected, selected_slices = subset_cache(data, slices, indices)
        destination = staging / split
        (destination / 'processed').mkdir(parents=True)
        (destination / 'raw').mkdir()
        torch.save((selected, selected_slices, data_class), destination / 'processed/data.pt')
        stats = calculate_stats(selected, selected_slices, template)
        (destination / 'processed/stats.json').write_text(json.dumps(stats, indent=2))
        (destination / 'raw/smiles.json').write_text(json.dumps([smiles[i] for i in indices]))
        for name in ['pre_transform.pt', 'pre_filter.pt']:
            source = root / 'processed' / name
            if source.exists():
                shutil.copy2(source, destination / 'processed' / name)
        checked = audit_cache(destination / 'processed/data.pt')
        if checked['errors']:
            raise ValueError(f'Staged split failed validation: {split}: {checked}')
    backup.mkdir(parents=True)
    manifest = {
        'seed': 0, 'sizes': {name: len(ids) for name, ids in split_indices.items()},
        'source_cache_sha256': file_digest(full_path),
        'source_smiles_sha256': file_digest(root / 'raw/smiles.json'),
        'indices': split_indices,
        'backup_directory': str(backup),
    }
    (backup / 'repair_manifest.json').write_text(json.dumps(manifest, indent=2))
    archived, installed = [], []
    try:
        for split in ['train', 'valid', 'test', 'val']:
            source = root / split
            if source.exists():
                source.rename(backup / split)
                archived.append(split)
        for split in split_indices:
            (staging / split).rename(root / split)
            installed.append(split)
    except Exception:
        for split in reversed(installed):
            (root / split).rename(staging / split)
        for split in reversed(archived):
            (backup / split).rename(root / split)
        raise
    print(f'Repaired {root}; previous splits and record indices: {backup}', flush=True)
    return {key: value for key, value in manifest.items() if key != 'indices'}


def repair_statistics(root):
    """Replace stale derived statistics only; preserve each previous JSON."""
    repaired = []
    backup = root / 'stats_backups' / f'audit_{time.time_ns()}'
    for split in ['', 'train', 'valid', 'test']:
        path = root / split / 'processed/data.pt'
        if not path.exists():
            continue
        data, slices, _ = load_cache(path)
        old_stats = read_json(path.with_name('stats.json'))
        new_stats = calculate_stats(data, slices, old_stats)
        stale = any(
            not np.allclose(old_stats['marginals'][key], values, atol=1e-7, rtol=1e-6)
            for key, values in new_stats['marginals'].items()
        )
        if not stale:
            continue
        backup.mkdir(parents=True, exist_ok=True)
        stats_path = path.with_name('stats.json')
        shutil.copy2(stats_path, backup / f'{split or "full"}_stats.json')
        temporary = stats_path.with_suffix('.json.audit_tmp')
        temporary.write_text(json.dumps(new_stats, indent=2))
        temporary.replace(stats_path)
        repaired.append(str(stats_path))
    return {'files': repaired, 'backup_directory': str(backup) if repaired else None}


class _LazyMolecule:
    """Keep RDKit's serialized bytes instead of materializing a million molecules."""

    __slots__ = ('binary',)

    def __init__(self, binary):
        self.binary = binary

    def __setstate__(self, state):
        # The cached molecules' Python state is only an empty property dict.
        if state != ({},):
            raise ValueError('Unexpected RDKit molecule state in raw cache')


class _MoleculeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'rdkit.Chem.rdchem' and name == 'Mol':
            return _LazyMolecule
        return super().find_class(module, name)


def audit_chemistry(root, limit=None):
    data, slices, _ = load_cache(root / 'processed/data.pt')
    raw_stats = read_json(root / 'raw/stats.json')
    with (root / 'raw/mols.pkl').open('rb') as stream:
        molecules = _MoleculeUnpickler(stream).load()
    size = len(slices['x']) - 1
    indices = list(range(size)) if limit is None else sorted(random.Random(0).sample(range(size), min(size, limit)))
    failures = Counter()
    examples = {}
    aux_values = raw_stats.get('auxiliary_node_state_values', {})
    atom_decoder, bond_decoder, charge_decoder = raw_stats['atom_types'], raw_stats['bond_types'], raw_stats['charges']
    labels = {key: classes(data[key]).numpy() for key in categorical_dims(read_json(root / 'processed/stats.json'))}
    edge_index = data['edge_index'].numpy()
    positions = data['node_pos'].numpy()
    node_slices, edge_slices = slices['x'].numpy(), slices['edge_index'].numpy()
    smiles = read_json(root / 'raw/smiles.json')
    properties_path = next((p for p in [root / 'raw/props.json', root / 'raw/props_mols.json'] if p.exists()), None)
    properties = read_json(properties_path) if properties_path else None

    def check(condition, kind, index):
        if not condition:
            failures[kind] += 1
            examples.setdefault(kind, index)

    check(len(molecules) == size, 'raw_molecule_count', -1)
    for step, index in enumerate(indices):
        molecule = Chem.Mol(molecules[index].binary)
        atoms = list(molecule.GetAtoms())
        start, end = node_slices[index:index + 2]
        first, last = edge_slices[index:index + 2]
        check(len(atoms) == end - start, 'atom_count', index)
        check([atom_decoder[int(v)] for v in labels['x'][start:end]] == [a.GetSymbol() for a in atoms], 'elements', index)
        check([charge_decoder[int(v)] for v in labels['node_charges'][start:end]] == [float(a.GetFormalCharge()) for a in atoms], 'charges', index)
        expected_edges = {}
        for bond in molecule.GetBonds():
            left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            expected_edges[left, right] = expected_edges[right, left] = str(bond.GetBondType())
        actual_edges = {(int(a), int(b)): bond_decoder[int(label)] for (a, b), label in zip(edge_index[:, first:last].T, labels['edge_attr'][first:last])}
        check(actual_edges == expected_edges, 'bonds', index)
        if molecule.GetNumConformers():
            raw_pos = molecule.GetConformer().GetPositions().astype(np.float32)
            raw_pos -= raw_pos.mean(0, keepdims=True)
            check(np.allclose(raw_pos, positions[start:end], atol=2e-5, rtol=1e-5), 'coordinates', index)
        else:
            failures['missing_raw_conformer'] += 1
        check(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False) == smiles[index], 'raw_smiles_alignment', index)
        if properties is not None and data['y'].shape[-1]:
            prop = properties[index]
            if isinstance(prop, dict):
                prop = list(prop.values())
            check(np.allclose(np.asarray(prop, dtype=np.float32), data['y'][index].numpy(), atol=1e-6, rtol=1e-6), 'targets', index)
        for name, vocabulary in aux_values.items():
            if name == 'node_mmff_type':
                mmff = ChemicalForceFields.MMFFGetMoleculeProperties(Chem.Mol(molecule))
                check(mmff is not None, 'mmff_properties', index)
                if mmff is None:
                    continue
                expected = [str(mmff.GetMMFFAtomType(i)) for i in range(len(atoms))]
            elif name == 'node_is_in_ring':
                expected = [a.IsInRing() for a in atoms]
            elif name == 'node_is_aromatic':
                expected = [a.GetIsAromatic() for a in atoms]
            elif name == 'node_hybridization':
                expected = [a.GetHybridization().name for a in atoms]
            else:
                raise ValueError(f'Unsupported auxiliary state: {name}')
            check([vocabulary[int(v)] for v in labels[name][start:end]] == expected, name, index)
        if (step + 1) % 10000 == 0:
            print(f'  {root.name}: chemistry {step + 1}/{len(indices)}', flush=True)
    return {'checked_molecules': len(indices), 'total_molecules': size, 'full_coverage': len(indices) == size,
            'failures': dict(failures), 'first_failure_indices': examples,
            'raw_molecules_sha256': file_digest(root / 'raw/mols.pkl')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='datasets', type=Path)
    parser.add_argument('--datasets', nargs='+', default=['qm9', 'qm9_mmff', 'qm9_atom_details', 'gdb13', 'zinc'])
    parser.add_argument('--output', type=Path, default=Path('outputs/dataset_audit/report.json'))
    parser.add_argument('--skip-chemistry', action='store_true')
    parser.add_argument('--repair-qm9-splits', action='store_true', help='Repair inconsistent QM9 splits, retaining the old directories in split_backups')
    parser.add_argument('--repair-statistics', action='store_true', help='Recompute stale marginals, retaining their previous JSON files')
    parser.add_argument('--other-chemistry-samples', type=int, default=1000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {}
    for name in args.datasets:
        root = args.root / name
        print(f'Auditing {root}', flush=True)
        entry = {'caches': {}}
        if args.repair_qm9_splits and name.startswith('qm9') and audit_splits(root)['errors']:
            entry['repair'] = repair_qm9_splits(root)
        if args.repair_statistics:
            entry['statistics_repair'] = repair_statistics(root)
        for split in ['', 'train', 'valid', 'val', 'test']:
            path = root / split / 'processed/data.pt'
            if path.exists():
                result = audit_cache(path)
                entry['caches'][split or 'full'] = result
                print(f"  {split or 'full'}: {result['graphs']} graphs, {len(result['errors'])} errors", flush=True)
        entry['splits'] = audit_splits(root)
        if not args.skip_chemistry:
            limit = None if name.startswith('qm9') else args.other_chemistry_samples
            entry['chemistry'] = audit_chemistry(root, limit)
        report[name] = entry
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"  split errors: {entry['splits']['errors']}", flush=True)
    failed = any(
        entry['splits']['errors'] or entry.get('chemistry', {}).get('failures')
        or any(cache['errors'] for cache in entry['caches'].values())
        for entry in report.values()
    )
    print(f'Report: {args.output}; result: {"FAIL" if failed else "PASS"}', flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
