"""Analytic checks of probability distances and transparent group aggregation."""
from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

import score_generated_distributions as scores
from compare_generated_distributions import CATEGORICAL_FIELDS


def aggregate(distance):
    counts = {'0': round(100 * (1 - distance)), '1': round(100 * distance)}
    aux = ['node_mmff_type', 'node_hybridization', 'node_is_in_ring', 'node_is_aromatic']
    return {'categorical': {**{field: {} for field in CATEGORICAL_FIELDS},
                            'dataset_specific_atom_features': {}},
            'counts': {field: dict(counts) for field in CATEGORICAL_FIELDS},
            'dataset_specific_atom_feature_counts': {field: dict(counts) for field in aux},
            'scalar_histograms': {field: dict(counts) for field in ['valid_num_atoms', 'valid_num_bonds',
                                                                  'valid_num_rings', 'valid_num_components']},
            'substructure_presence': {'group_A': distance, 'group_B': distance},
            'num_input_graphs': 100, 'num_valid_molecules': 100, 'valid_fraction': 1.,
            'uniqueness_fraction': .8, 'mmff_coverage': {'molecules': 100, 'atoms': 200,
                                                     'failed_valid_molecules': 0, 'valid_molecule_fraction': 1.}}


def fake_report():
    return {'schema_version': 2, 'metadata': {'reference_dataset': 'qm9', 'reference_split': 'test',
                                            'expected_seeds': [0, 1]}, 'reference': aggregate(0),
            'representations': {'base': {'runs': [{'seed': 0, 'generated': aggregate(0)},
                                                 {'seed': 1, 'generated': aggregate(.2)}]},
                                'mmff': {'runs': [{'seed': 0, 'generated': aggregate(.08)},
                                                  {'seed': 1, 'generated': aggregate(.1)}]}}}


class DistributionScoreTests(unittest.TestCase):
    def test_distances_have_analytic_extreme_and_transport_values(self):
        same = scores.distribution_metrics({'A': 1}, {'A': 5})
        self.assertEqual(same['tvd'], 0)
        self.assertEqual(same['js_distance'], 0)
        disjoint = scores.distribution_metrics({'0': 1}, {'2': 1}, numeric=True)
        self.assertEqual(disjoint['tvd'], 1)
        self.assertEqual(disjoint['js_distance'], 1)
        self.assertEqual(disjoint['wasserstein'], 2)
        mixed = scores.distribution_metrics({'0': 3, '1': 1}, {'0': 1, '1': 1}, numeric=True)
        self.assertAlmostEqual(mixed['tvd'], .25)
        self.assertAlmostEqual(mixed['wasserstein'], .25)

    def test_presence_uses_independent_bernoulli_distributions(self):
        metric = scores.presence_metrics({'alcohol': .8, 'amine': .7}, {'alcohol': .4, 'amine': .3})
        self.assertAlmostEqual(metric['tvd'], .4)
        self.assertGreater(metric['js_distance'], 0)
        with self.assertRaises(ValueError):
            scores.presence_metrics({'alcohol': .5}, {'amine': .5})

    def test_union_support_includes_unseen_categories_and_validates_input(self):
        metric = scores.distribution_metrics({'C': 1}, {'C': 1, 'O': 1})
        self.assertAlmostEqual(metric['tvd'], .5)
        with self.assertRaises(ValueError):
            scores.distribution_metrics({'C': -1}, {'C': 1})
        self.assertIsNone(scores.distribution_metrics({}, {'C': 1})['tvd'])

    def test_each_group_has_equal_weight_regardless_of_descriptor_count(self):
        metrics = {field: {'tvd': 0.} for fields in scores.GROUPS.values() for field in fields}
        metrics['atom_symbols']['tvd'] = 1.
        value, group_values, missing = scores.composite_score(metrics, dict.fromkeys(scores.GROUPS, 1.), 'tvd')
        self.assertEqual(missing, [])
        self.assertAlmostEqual(group_values['composition'], 1 / 3)
        self.assertAlmostEqual(value, 1 / 15)

    def test_undefined_active_group_prevents_incomparable_score(self):
        metrics = {field: {'tvd': 0.} for fields in scores.GROUPS.values() for field in fields}
        metrics['node_mmff_type']['tvd'] = None
        weights = dict.fromkeys(scores.GROUPS, 1.)
        value, _, missing = scores.composite_score(metrics, weights, 'tvd')
        self.assertIsNone(value)
        self.assertEqual(missing, ['atomic_environments'])
        weights['atomic_environments'] = 0
        self.assertEqual(scores.composite_score(metrics, weights, 'tvd')[0], 0.)

    def test_score_then_seed_summary_preserves_covariance_and_ranking(self):
        result = scores.analyze_report(fake_report(), dict.fromkeys(scores.GROUPS, 1.))
        stat = result['representations']['base']['summary']['macro_tvd']
        self.assertAlmostEqual(stat['mean'], .1)
        self.assertAlmostEqual(stat['std'], math.sqrt(.02))
        self.assertEqual(stat['n'], 2)
        self.assertEqual(result['ranking_macro_tvd'], ['mmff', 'base'])
        self.assertEqual(result['representations']['atom_details']['missing_seeds'], [0, 1])
        self.assertTrue(any('atom_details' in warning for warning in result['warnings']))

    def test_zinc_scores_and_inference_exclude_mmff_but_keep_mmff_descriptors(self):
        from compare_distribution_statistics import analyze_inputs
        for dataset in ('zinc', 'zinc250k', 'zinc_atom_details', 'zinc250k_atom_details'):
            with self.subTest(dataset=dataset):
                report = fake_report()
                report['metadata']['reference_dataset'] = dataset
                report['representations']['atom_details'] = deepcopy(report['representations']['mmff'])
                result = scores.analyze_report(report, dict.fromkeys(scores.GROUPS, 1.))
                self.assertEqual(set(result['representations']), {'base', 'atom_details'})
                self.assertNotIn('mmff', result['ranking_macro_tvd'])
                run = result['representations']['base']['runs'][0]
                self.assertIsNotNone(run['descriptor_metrics']['node_mmff_type']['tvd'])
                self.assertIsNotNone(run['macro_tvd'])
                header = next(line for line in scores.wasserstein_markdown_lines(result)
                              if line.startswith('| Descrittore'))
                self.assertNotIn('MMFF', header)
                inference = analyze_inputs({'test': result})
                primary = [row for row in inference['tests'] if row['family'] == 'primary']
                self.assertEqual(len(primary), 1)
                self.assertEqual(primary[0]['comparison'], 'pairwise')
                self.assertEqual({primary[0]['model_a'], primary[0]['model_b']}, {'base', 'atom_details'})

    def test_single_seed_has_undefined_std_and_duplicate_seed_fails(self):
        report = fake_report()
        report['representations']['base']['runs'].pop()
        result = scores.analyze_report(report, dict.fromkeys(scores.GROUPS, 1.))
        self.assertIsNone(result['representations']['base']['summary']['macro_tvd']['std'])
        report['representations']['mmff']['runs'][1]['seed'] = 0
        with self.assertRaises(ValueError):
            scores.analyze_report(report, dict.fromkeys(scores.GROUPS, 1.))

    def test_custom_weights_and_source_report_are_preserved(self):
        report = fake_report()
        original = deepcopy(report)
        weights = dict.fromkeys(scores.GROUPS, 0.)
        weights['bonds'] = 2.
        result = scores.analyze_report(report, weights)
        self.assertEqual(result['score_definition']['normalized_group_weights']['bonds'], 1.)
        self.assertEqual(report, original)
        weights['bonds'] = -1
        with self.assertRaises(ValueError):
            scores.analyze_report(report, weights)

    def test_plot_axes_stay_canonical_when_rankings_and_json_order_differ(self):
        report = fake_report()
        report['representations']['atom_details'] = deepcopy(report['representations']['base'])
        for run in report['representations']['atom_details']['runs']:
            run['generated'] = aggregate(.4)
        result = scores.analyze_report(report, dict.fromkeys(scores.GROUPS, 1.))
        # Deliberately make insertion order and ranking differ from display order.
        result['representations'] = {name: result['representations'][name]
                                     for name in ['atom_details', 'mmff', 'base']}
        self.assertEqual(result['ranking_macro_tvd'], ['mmff', 'base', 'atom_details'])
        original = deepcopy(result)
        figures = {}
        def capture(figure, directory, name):
            figures[name] = figure
        with tempfile.TemporaryDirectory() as directory, patch.object(scores, 'save_figure', side_effect=capture):
            try:
                scores.draw_plots(result, Path(directory))
                scores.write_wasserstein_outputs(result, Path(directory))
                for name in ['groups_tvd', 'descriptors_tvd']:
                    ax = figures[name].axes[0]
                    self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()],
                                     ['Base', 'MMFF', 'Atom details'])
                matrix = np.asarray(figures['groups_tvd'].axes[0].images[0].get_array())
                expected = [[result['representations'][name]['summary']['group_tvd'][group]['mean']
                             for name in ['base', 'mmff', 'atom_details']] for group in scores.GROUPS]
                np.testing.assert_allclose(matrix, expected)
                for ax in figures['ranking'].axes:
                    self.assertEqual([tick.get_text().split(' (n=')[0] for tick in ax.get_yticklabels()],
                                     ['Base', 'MMFF', 'Atom details'])
                for ax in figures['wasserstein'].axes:
                    if ax.get_visible():
                        self.assertEqual([tick.get_text() for tick in ax.get_yticklabels()],
                                         ['Base', 'MMFF', 'Atom details'])
                self.assertEqual(result, original)
            finally:
                for figure in figures.values():
                    plt.close(figure)

    def test_plot_order_respects_zinc_and_missing_representations(self):
        result = {'metadata': {'reference_dataset': 'zinc'},
                  'representations': {'atom_details': {}, 'mmff': {}, 'base': {}}}
        self.assertEqual(scores.plot_representations(result), ['base', 'atom_details'])
        result['metadata']['reference_dataset'] = 'qm9'
        del result['representations']['mmff']
        self.assertEqual(scores.plot_representations(result), ['base', 'atom_details'])


if __name__ == '__main__':
    unittest.main()
