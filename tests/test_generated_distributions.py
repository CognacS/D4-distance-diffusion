"""Checks of seed discovery and the chemical distributions used for model comparison."""
import json
import io
from contextlib import redirect_stdout, redirect_stderr
import math
import os
import pickle
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from rdkit import Chem
from torch_geometric.data import InMemoryDataset

import compare_generated_distributions as analysis
from src.datatypes.sparse import SparseGraph


STATS = {'atom_types': ['C', 'F', 'N', 'O'], 'charges': [-1., 0., 1.],
         'bond_types': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']}


def graph_from_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    atoms = torch.tensor([STATS['atom_types'].index(a.GetSymbol()) for a in molecule.GetAtoms()])
    charges = torch.tensor([STATS['charges'].index(float(a.GetFormalCharge())) for a in molecule.GetAtoms()])
    pairs, kinds = [], []
    for bond in molecule.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pairs.extend([[a, b], [b, a]])
        kinds.extend([STATS['bond_types'].index(bond.GetBondType().name)] * 2)
    return SparseGraph(x=atoms, node_charges=charges,
                       edge_index=torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T.contiguous(),
                       edge_attr=torch.tensor(kinds, dtype=torch.long))


class DistributionAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir='/tmp')
        self.root = Path(self.directory.name)
        path = self.root / 'stats.json'
        path.write_text(json.dumps(STATS))
        self.decoder = analysis.ChemistryDecoder(path)

    def tearDown(self):
        self.directory.cleanup()

    def make_run(self, name, seed, timestamp=1):
        run = self.root / 'checkpoints/experiment' / name
        run.mkdir(parents=True)
        file = run / 'generated_graphs.pkl'
        file.touch()
        os.utime(file, (timestamp, timestamp))
        if seed is not None:
            metadata_dir = self.root / 'checkpoints/wandb' / ('run-20261019-' + name) / 'files'
            metadata_dir.mkdir(parents=True)
            (metadata_dir / 'wandb-metadata.json').write_text(json.dumps({'args': [f'seed={seed}']}))
        return file

    def aggregate(self, smiles):
        return analysis.aggregate_graphs([graph_from_smiles(item) for item in smiles], self.decoder, desc='test')

    def test_discovery_uses_metadata_and_reports_unknown_seeds(self):
        self.make_run('v0_seedtwo', 2)
        self.make_run('v2_unknown', None)
        messages = []
        runs = analysis.discover_runs([self.root / 'checkpoints/experiment'], self.root / 'checkpoints', [0, 1, 2], messages)
        self.assertEqual([run['seed'] for run in runs], [2])
        self.assertTrue(any('unknown' in message for message in messages))

    def test_discovery_selects_one_latest_generation_per_seed(self):
        old = self.make_run('v0_old', 0, 1)
        new = self.make_run('v0_new', 0, 2)
        newer = new.with_name('generated_graphs_new.pkl')
        newer.touch()
        os.utime(newer, (3, 3))
        messages = []
        runs = analysis.discover_runs([old.parent.parent], self.root / 'checkpoints', [0, 1, 2], messages)
        self.assertEqual(len(runs), 1)
        self.assertEqual(Path(runs[0]['generated_file']), newer)
        self.assertEqual(len(messages), 2)

    def test_metadata_disagreement_fails(self):
        file = self.make_run('v0_conflict', 0)
        (file.parent / 'config.yaml').write_text('seed: 1\n')
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            analysis.read_run_metadata(file.parent, self.root / 'checkpoints')

    def test_cached_reference_can_be_read_without_raw_molecules_or_preprocessing(self):
        (self.root / 'raw').mkdir()
        (self.root / 'raw/stats.json').write_text(json.dumps(STATS))
        folder = self.root / 'test/processed'
        folder.mkdir(parents=True)
        graphs = [graph_from_smiles('CCO'), graph_from_smiles('CC[O-]')]
        InMemoryDataset.save(graphs, str(folder / 'data.pt'))
        dataset = analysis.load_reference_cache(self.root, 'test')
        self.assertEqual(len(dataset), 2)
        self.assertTrue(torch.equal(dataset[1].node_charges, graphs[1].node_charges))

    def test_compact_pickle_loader_preserves_tensor_views(self):
        storage = torch.arange(3000)
        graphs = [SparseGraph(x=storage[index:index + 3]) for index in range(12)]
        for protocol in [3, 4, 5]:
            path = self.root / f'graphs{protocol}.pkl'
            with path.open('wb') as stream:
                pickle.dump(graphs, stream, protocol=protocol)
            restored = analysis.load_graph_collection(path)
            self.assertEqual(len(restored), len(graphs))
            for actual, expected in zip(restored, graphs):
                self.assertTrue(torch.equal(actual.x, expected.x))
            self.assertEqual(restored[0].x.untyped_storage().data_ptr(),
                             restored[1].x.untyped_storage().data_ptr())

    def test_all_auxiliary_profiles_are_recomputed_and_charges_preserved(self):
        graph = graph_from_smiles('CC[O-]')
        graph.node_mmff_type = torch.tensor([999, 999, 999])
        graph.node_is_in_ring = torch.ones(3)
        molecule = analysis.canonicalize_largest_fragment(self.decoder.graph_to_molecule(graph))['analysis_mol']
        profile = analysis.build_molecule_profile(molecule)
        self.assertEqual(profile['molecular_charge'], ['-1'])
        self.assertEqual(profile['dataset_specific_atom_features']['node_is_in_ring'], ['False'] * 3)
        self.assertEqual(len(profile['dataset_specific_atom_features']['node_mmff_type']), 3)
        self.assertNotIn('999', profile['dataset_specific_atom_features']['node_mmff_type'])
        self.assertIn('O|charge=-1', profile['element_charge'])
        self.assertTrue(any('charge=-1' in token for token in profile['mmff_charge']))

    def test_denominators_keep_duplicates_and_count_atoms_and_molecules_separately(self):
        result = self.aggregate(['CCO', 'CCO', 'CC[O-]'])
        self.assertEqual(result['num_valid_molecules'], 3)
        self.assertEqual(result['num_unique_valid_smiles'], 2)
        self.assertEqual(result['categorical_denominators']['formal_charges'], 9)
        self.assertEqual(result['categorical_denominators']['molecular_charge'], 3)
        self.assertEqual(result['scalars']['valid_num_atoms']['count'], 3)
        self.assertEqual(result['counts']['molecular_charge'], {'0': 2, '-1': 1})
        self.assertAlmostEqual(sum(result['categorical']['formal_charges'].values()), 1.)

    def test_failed_mmff_is_undefined_and_reported(self):
        with patch.object(analysis, 'derive_mmff_atom_types', return_value=[]):
            result = self.aggregate(['CCO'])
        self.assertEqual(result['mmff_coverage']['failed_valid_molecules'], 1)
        self.assertEqual(result['categorical']['mmff_charge'], {})
        self.assertIsNone(analysis.compare_aggregates(result, result)['categorical_tvd']['mmff_charge'])

    def test_distances_have_known_values(self):
        self.assertEqual(analysis.total_variation_distance(analysis.Counter({'a': 3}), analysis.Counter({'b': 2})), 1.)
        self.assertEqual(analysis.discrete_wasserstein([0, 2], [1, 3]), 1.)
        self.assertEqual(analysis.discrete_wasserstein([1, 2], [1, 2]), 0.)

    def test_seed_summary_uses_zero_for_absent_categories_and_sample_std(self):
        result = analysis.summarize_tree([{'C': .8, 'O': .2}, {'C': 1.}], True)
        self.assertAlmostEqual(result['O']['mean'], .1)
        self.assertAlmostEqual(result['O']['std'], math.sqrt(.02))
        self.assertEqual(result['O']['n'], 2)
        self.assertIsNone(analysis.numeric_summary([1.])['std'])

    def test_empty_chemical_distributions_are_excluded_from_seed_mean(self):
        valid = self.aggregate(['CCO'])
        with patch.object(analysis, 'derive_mmff_atom_types', return_value=[]):
            unsupported = self.aggregate(['CCO'])
        runs = [{'generated': analysis.strip_internal_fields(item), 'comparison': analysis.compare_aggregates(valid, item)}
                for item in [valid, unsupported]]
        summary = analysis.summarize_runs(runs)
        self.assertEqual(summary['num_seeds'], 2)
        self.assertTrue(all(stat['n'] == 1 for stat in summary['distributions']['node_mmff_type'].values()))

    def test_split_aliases_and_reproducible_subsampling(self):
        self.assertEqual(analysis.normalize_split('validation'), 'valid')
        self.assertEqual(analysis.normalize_split('val'), 'valid')
        self.assertEqual(analysis.normalize_split('train'), 'train')
        self.assertEqual(analysis.select_indices(100, 10, 7), analysis.select_indices(100, 10, 7))
        with self.assertRaises(ValueError):
            analysis.normalize_split('not_a_split')

    def test_cli_compares_three_representations_and_records_missing_seeds(self):
        datasets = {}
        graphs = [graph_from_smiles('CCO'), graph_from_smiles('CC[O-]')]
        for suffix in ['', '_mmff', '_atom_details']:
            root = self.root / ('qm9' + suffix)
            (root / 'raw').mkdir(parents=True)
            (root / 'raw/stats.json').write_text(json.dumps(STATS))
            datasets['qm9' + suffix] = root
        reference = datasets['qm9']
        (reference / 'valid/processed').mkdir(parents=True)
        InMemoryDataset.save(graphs, str(reference / 'valid/processed/data.pt'))
        arguments = ['compare_generated_distributions.py', '--reference-root', str(reference),
                     '--reference-split', 'validation', '--checkpoints-root', str(self.root / 'checkpoints'),
                     '--output', str(self.root / 'result/report.json')]
        for name in ['base', 'mmff', 'atom_details']:
            file = self.make_run('v0_' + name, 0)
            with file.open('wb') as stream:
                pickle.dump(graphs, stream)
            arguments.extend(['--generated', f'{name}={file}'])
        snapshots = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        with patch.object(sys, 'argv', arguments), \
                patch.object(analysis, 'resolve_dataset_root', side_effect=lambda name, override: str(datasets[name])), \
                patch.object(analysis, 'write_comparison_plots', return_value=[]), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            analysis.main()
        report = json.loads((self.root / 'result/report.json').read_text())
        self.assertEqual(report['metadata']['reference_split'], 'valid')
        self.assertEqual(len(report['warnings']), 3)
        for name, entry in report['representations'].items():
            self.assertEqual(entry['missing_seeds'], [1, 2])
            self.assertEqual(entry['available_seeds'], [0])
            self.assertEqual(entry['summary']['num_input_graphs'], 2)
            self.assertEqual(entry['summary']['comparison']['categorical_tvd']['formal_charges']['mean'], 0.)
            self.assertEqual(entry['runs'][0]['generated']['mmff_coverage']['molecules'], 2)
        for path, original in snapshots.items():
            self.assertEqual(path.read_bytes(), original, str(path))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
