"""Analytic and reference checks of seed-level permutation inference."""
import unittest

import numpy as np
from scipy.stats import permutation_test

from compare_distribution_statistics import holm_adjust, permutation_comparison, analyze_inputs
from score_generated_distributions import GROUPS


def score_report():
    representations = {}
    for name, values in [('base', [.1, .2, .3]), ('mmff', [.4, .5, .6]), ('atom_details', [.7, .8, .9])]:
        representations[name] = {'runs': [
            {'seed': index, 'macro_tvd': value, 'macro_js_distance': value,
             'group_tvd': dict.fromkeys(GROUPS, value),
             'descriptor_metrics': {'valid_num_atoms': {'wasserstein': value}}}
            for index, value in enumerate(values)]}
    return {'metadata': {'reference_dataset': 'qm9', 'reference_split': 'test'},
            'score_definition': {'group_weights': dict.fromkeys(GROUPS, 1)},
            'representations': representations}


class PermutationInferenceTests(unittest.TestCase):
    def test_three_vs_three_exact_bilateral_floor_is_point_one(self):
        result = permutation_comparison([[0, 0, 0], [1, 1, 1]])
        self.assertEqual(result['n_permutations'], 20)
        self.assertEqual(result['method'], 'exact_permutation')
        self.assertEqual(result['p_raw'], .1)
        self.assertEqual(result['p_resolution_lower_bound'], .1)

    def test_matches_scipy_exact_two_sample_test(self):
        a, b = [.001, .004, .009], [.002, .003, .008]
        reference = permutation_test((a, b), lambda x, y: np.mean(x) - np.mean(y),
                                     vectorized=False, n_resamples=np.inf).pvalue
        self.assertAlmostEqual(permutation_comparison([a, b])['p_raw'], reference)

    def test_omnibus_constant_and_perfect_separation_are_handled(self):
        constant = permutation_comparison([[.2] * 3] * 3, omnibus=True)
        self.assertEqual(constant['statistic'], 0.)
        self.assertEqual(constant['p_raw'], 1.)
        separated = permutation_comparison([[0] * 3, [1] * 3, [2] * 3], omnibus=True)
        self.assertEqual(separated['n_permutations'], 1680)
        self.assertEqual(separated['statistic'], 'infinity')
        self.assertAlmostEqual(separated['p_raw'], 6 / 1680)

    def test_holm_adjustment_monotone_and_in_original_order(self):
        result = holm_adjust([.01, .04, .03])
        for actual, expected in zip(result, [.03, .06, .06]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(holm_adjust([]), [])

    def test_monte_carlo_is_reproducible_and_never_reports_zero(self):
        a, b = list(range(6)), list(range(20, 26))
        first = permutation_comparison([a, b], max_permutations=50, rng_seed=7)
        self.assertEqual(first, permutation_comparison([a, b], max_permutations=50, rng_seed=7))
        self.assertEqual(first['method'], 'monte_carlo_permutation')
        self.assertGreaterEqual(first['p_raw'], 1 / 51)

    def test_families_span_all_references_and_models_are_unpaired(self):
        result = analyze_inputs({'train': score_report(), 'test': score_report()})
        primary = [row for row in result['tests'] if row['family'] == 'primary']
        self.assertEqual(len(primary), 8)
        self.assertTrue(all(row['holm_family_size'] == 8 for row in primary))
        pairs = [row for row in primary if row['comparison'] == 'pairwise']
        self.assertTrue(all(row['n_permutations'] == 20 for row in pairs))
        self.assertTrue(all(not row['reject_at_alpha'] for row in pairs))
        self.assertTrue(all(row['p_holm'] >= row['p_raw'] for row in result['tests']))

    def test_insufficient_or_nonfinite_replicates_are_not_tested(self):
        with self.assertRaises(ValueError):
            permutation_comparison([[0.], [1., 2.]])
        with self.assertRaises(ValueError):
            permutation_comparison([[0., np.nan], [1., 2.]])


if __name__ == '__main__':
    unittest.main()
