"""Statistical checks for the dedicated four-population plotter."""
import math
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.container import BarContainer

import compare_three_representations as joint


class JointComparisonTests(unittest.TestCase):
    def groups(self):
        return {
            'base': [{'seed': 0, 'distributions': {'test': {'A': .2, 'B': .8}}},
                     {'seed': 2, 'distributions': {'test': {'A': .4, 'B': .6}}}],
            'mmff': [{'seed': 1, 'distributions': {'test': {'A': .1, 'B': .9}}}],
            'atom_details': [{'seed': 0, 'distributions': {'test': {'A': .7, 'B': .3}}}],
        }

    def test_seed_mean_standard_deviation_and_percentage_point_difference(self):
        rows = joint.comparison_table('test', {'test': {'A': .2, 'B': .8}}, self.groups(), 20)
        row = next(row for row in rows if row['category'] == 'A')
        self.assertAlmostEqual(row['base']['mean'], .3)
        self.assertAlmostEqual(row['base']['std'], math.sqrt(.02))
        self.assertAlmostEqual(row['base']['difference_percentage_points'], 10.)
        self.assertEqual(row['base']['seeds'], [0, 2])
        self.assertIsNone(row['mmff']['std'])

    def test_every_category_has_four_real_bars_on_the_same_axes(self):
        rows = joint.comparison_table('test', {'test': {'A': .2, 'B': .8}}, self.groups(), 20)
        figure = joint.plot_descriptor('test', rows, 'Test')
        top, difference = figure.axes
        bars = [container for container in top.containers if isinstance(container, BarContainer)]
        self.assertEqual(len(bars), 4)
        self.assertTrue(all(len(container) == 2 for container in bars))
        self.assertEqual(len([c for c in difference.containers if isinstance(c, BarContainer)]), 3)
        self.assertEqual(len(top.get_legend().get_texts()), 4)
        plt.close(figure)

    def test_missing_representation_is_undefined_not_zero(self):
        groups = self.groups()
        groups['atom_details'] = []
        rows = joint.comparison_table('test', {'test': {'A': .2, 'B': .8}}, groups, 20)
        self.assertTrue(all(row['atom_details']['mean'] is None for row in rows))
        self.assertTrue(all(row['atom_details']['n'] == 0 for row in rows))
        self.assertIn('non disponibile', joint.legend_handles(rows)[-1].get_label())

    def test_other_category_preserves_total_probability_per_seed(self):
        groups = self.groups()
        rows = joint.comparison_table('test', {'test': {'A': .2, 'B': .8}}, groups, 1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]['category'], 'Altre categorie')
        self.assertAlmostEqual(sum(row['dataset'] for row in rows), 1.)
        for name in joint.REPRESENTATIONS:
            self.assertAlmostEqual(sum(row[name]['mean'] for row in rows), 1.)
        presence_groups = {name: [{'seed': 0, 'distributions': {'substructure_presence': {'A': .7, 'B': .8}}}]
                           for name in joint.REPRESENTATIONS}
        rows = joint.comparison_table('substructure_presence', {'substructure_presence': {'A': .7, 'B': .8}},
                                      presence_groups, 1)
        self.assertEqual(len(rows), 1)
        self.assertNotEqual(rows[0]['category'], 'Altre categorie')

    def test_saved_exact_counts_take_precedence_over_rounded_probabilities(self):
        aggregate = {'categorical': {name: {} for name in joint.analysis.CATEGORICAL_FIELDS},
                     'counts': {'atom_symbols': {'C': 2, 'O': 1}}, 'substructure_presence': {}}
        aggregate['categorical']['dataset_specific_atom_features'] = {}
        aggregate['categorical']['atom_symbols'] = {'C': .666667, 'O': .333333}
        result = joint.distributions(aggregate)
        self.assertEqual(result['atom_symbols']['C'], 2 / 3)

    def test_new_analysis_forwards_dataset_and_split_and_uses_a_separate_report(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as directory:
            output = Path(directory) / 'new_analysis'
            captured = {}

            def run_analysis():
                options = joint.analysis.parse_args()
                captured['dataset'] = options.reference_dataset
                captured['split'] = options.reference_split
                options.output.parent.mkdir(parents=True)
                options.output.write_text(json.dumps({'metadata': {'reference_dataset': 'gdb13', 'reference_split': 'valid'}}))

            arguments = ['compare_three_representations.py', '--reference-dataset', 'gdb13',
                         '--reference-split', 'valid', '--output-dir', str(output)]
            with patch.object(sys, 'argv', arguments), \
                    patch.object(joint.analysis, 'main', side_effect=run_analysis), \
                    patch.object(joint, 'make_plots', return_value={}), redirect_stdout(io.StringIO()):
                joint.main()
                self.assertEqual(sys.argv, arguments)
            self.assertEqual(captured, {'dataset': 'gdb13', 'split': 'valid'})
            self.assertTrue((output / 'analysis/report.json').is_file())
            result = json.loads((output / 'joint_comparison.json').read_text())
            self.assertEqual(result['source_report'], str(output / 'analysis/report.json'))


if __name__ == '__main__':
    unittest.main()
