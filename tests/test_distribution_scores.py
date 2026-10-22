"""Analytic checks of probability distances and transparent group aggregation."""
from copy import deepcopy
import math
import unittest

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


if __name__ == '__main__':
    unittest.main()
