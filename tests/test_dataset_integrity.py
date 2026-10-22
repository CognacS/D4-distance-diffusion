"""Regression checks for cached partitions and auxiliary categorical states."""

import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset
from omegaconf import OmegaConf

import src.data.datasets as datasets
from src.data.datasets.split import get_split_sizes, random_split_dataset, validate_cached_splits
from src.data.utils.graphs import get_torch_graphs_stats
from src.datatypes.dense import DenseGraph
from src.datatypes.sparse import SparseGraph
from src.models.d4.model import to_onehot_data
from src.models.midi.diffusion_model import MixedGraphSpatialDenoisingDiffusionModel
from src.models.midi.diffusion.noise_model import MarginalUniformTransition
from src.models.midi.metrics.abstract_metrics import SumExceptBatchKL
from src.models.midi.utils import PlaceHolder

from audit_preprocessed_datasets import load_cache, subset_cache


class RecordingDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.writes = 0

    def __len__(self):
        return self.size

    def subset_from(self, indices, name):
        self.writes += 1
        return indices


class DatasetIntegrityTests(unittest.TestCase):
    def test_invalid_partition_fails_before_writing(self):
        dataset = RecordingDataset(130194)
        with self.assertRaises(ValueError):
            random_split_dataset(dataset, {'train': 0.8, 'val': 0.2, 'test': 0.2})
        self.assertEqual(dataset.writes, 0)

    def test_fractional_partition_keeps_rounding_remainder(self):
        self.assertEqual(get_split_sizes(11, {'train': 0.8, 'test': 0.2}), {'train': 8, 'test': 3})

    def test_qm9_partition_is_complete_disjoint_and_reproducible(self):
        requested = {'train': None, 'valid': 10000, 'test': 10000}
        self.assertEqual(get_split_sizes(130194, requested), {'valid': 10000, 'test': 10000, 'train': 110194})
        state = random.getstate()
        first = random_split_dataset(RecordingDataset(130194), requested)
        second = random_split_dataset(RecordingDataset(130194), requested)
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), state)
        combined = [index for values in first.values() for index in values[0]]
        self.assertEqual(len(combined), 130194)
        self.assertEqual(set(combined), set(range(130194)))

    def test_existing_wrong_partition_is_rejected(self):
        cached = {'train': [range(104155), range(104155)],
                  'valid': [range(10000), range(10000)], 'test': [range(1), range(1)]}
        with self.assertRaisesRegex(ValueError, 'Cached split'):
            validate_cached_splits(cached, {'train': None, 'valid': 10000, 'test': 10000}, 130194)

    def test_qm9_resource_defaults_match_presets(self):
        for name in ['qm9', 'qm9_mmff', 'qm9_atom_details']:
            resource = datasets.reg_dataresources.get_instance(name, random_splits=None)
            self.assertEqual(resource.random_splits, {'train': None, 'valid': 10000, 'test': 10000})

    def test_single_class_is_one_hot_and_distances_remain_continuous(self):
        distances = torch.rand(2, 5, 5)
        graph = DenseGraph(x=torch.zeros(2, 5, dtype=torch.long),
                           node_charges=torch.zeros(2, 5, dtype=torch.long),
                           node_is_aromatic=torch.zeros(2, 5, dtype=torch.long),
                           edge_dist=distances)
        to_onehot_data(graph, x=4, node_charges=1, node_is_aromatic=1, edge_dist=1)
        self.assertEqual(graph.node_charges.shape, (2, 5, 1))
        self.assertTrue(torch.equal(graph.node_is_aromatic, torch.ones(2, 5, 1)))
        self.assertIs(graph.edge_dist, distances)
        # Reapplying conversion should not add a second feature axis.
        to_onehot_data(graph, x=4, node_charges=1, node_is_aromatic=1, edge_dist=1)
        self.assertEqual(graph.node_is_aromatic.shape, (2, 5, 1))

    def test_vectorized_cache_selection_matches_pyg_collation(self):
        graphs = []
        for nodes in [1, 3, 2]:
            edges = torch.tensor([[0, 1], [1, 0]]) if nodes > 1 else torch.empty(2, 0, dtype=torch.long)
            graphs.append(SparseGraph(x=torch.arange(nodes), edge_index=edges,
                                      edge_attr=torch.zeros(edges.shape[1], dtype=torch.long),
                                      node_pos=torch.zeros(nodes, 3), y=torch.arange(19)))
        data, slices = InMemoryDataset.collate(graphs)
        selected, selected_slices = subset_cache(data.to_dict(), slices, [2, 0])
        expected, expected_slices = InMemoryDataset.collate([graphs[2], graphs[0]])
        for key in expected.keys():
            self.assertTrue(torch.equal(selected[key], expected[key]), key)
            self.assertTrue(torch.equal(selected_slices[key], expected_slices[key]), key)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'single.pt'
            InMemoryDataset.save([graphs[0]], str(path))
            single, single_slices, _ = load_cache(path)
            self.assertEqual(single['y'].shape, (1, 19))
            self.assertEqual(single_slices['x'].tolist(), [0, 1])

    def test_large_nonedge_counts_keep_normalized_marginals(self):
        edges = torch.tensor([[0, 1], [1, 0]])
        graphs = [SparseGraph(x=torch.zeros(nodes, dtype=torch.long), edge_index=edges,
                              edge_attr=torch.zeros(2, dtype=torch.long)) for nodes in [10000, 2]]
        stats = get_torch_graphs_stats(graphs, num_classes={'x': 1, 'edge_attr': 1})
        possible_edges = 10000 * 9999 + 2
        self.assertAlmostEqual(stats['marginals']['edge_attr'][0], (possible_edges - 4) / possible_edges, places=14)
        self.assertAlmostEqual(sum(stats['marginals']['edge_attr']), 1., places=14)

    def test_midi_likelihood_sums_auxiliary_kls(self):
        torch.manual_seed(0)
        cfg = OmegaConf.create({'model': {
            'nu': {'p': 2.5, 'x': 1, 'c': 1, 'e': 1.5, 'y': 1, 'auxiliary_node_states': 1},
            'diffusion_noise_schedule': 'cosine', 'diffusion_steps': 100,
        }})
        marginals = {'node_is_in_ring': torch.tensor([0.5, 0.5]),
                     'node_is_aromatic': torch.tensor([0.5, 0.5]),
                     'node_hybridization': torch.tensor([0.5, 0.5])}
        noise = MarginalUniformTransition(cfg, torch.tensor([0.5, 0.5]), torch.tensor([0.5, 0.5]),
                                          torch.tensor([0.5, 0.5]), y_classes=0,
                                          auxiliary_node_state_marginals=marginals)
        clean = PlaceHolder(X=torch.tensor([[[1., 0.], [0., 1.]]]),
                            charges=torch.tensor([[[1., 0.], [1., 0.]]]),
                            E=torch.tensor([[[[0., 0.], [0., 1.]], [[0., 1.], [0., 0.]]]]),
                            y=torch.empty(1, 0), pos=torch.tensor([[[-1., 0., 0.], [1., 0., 0.]]]),
                            node_mask=torch.ones(1, 2, dtype=torch.bool),
                            auxiliary_node_states={name: torch.tensor([[[1., 0.], [1., 0.]]]) for name in marginals})
        noisy = noise.apply_noise(clean)
        pred = clean.copy()
        pred.auxiliary_node_states = {name: torch.tensor([[[0., 1.], [0., 1.]]]) for name in marginals}
        metrics = {name: SumExceptBatchKL() for name in marginals}
        model = SimpleNamespace(T=noise.T, noise_model=noise, auxiliary_node_state_names=list(marginals),
                                val_auxiliary_node_state_metrics=metrics,
                                val_metrics=lambda *_: {key: torch.tensor(0.) for key in ['PosMSE', 'XKl', 'ChargesKl', 'EKl']})
        value = MixedGraphSpatialDenoisingDiffusionModel.compute_Lt(
            model, clean, pred, noisy, noisy.t_int - 1, clean.node_mask, test=False)
        expected = noise.T * sum(metric.compute() for metric in metrics.values())
        self.assertGreater(float(expected), 0.)
        self.assertTrue(torch.allclose(value, expected))


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
