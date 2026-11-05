"""Exercise orchestration and public reports without running molecular analysis."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import compare_distributions as workflow


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fake_step(command, **kwargs):
    """Create tiny stage artifacts; never import chemistry/training modules."""
    script = Path(command[1]).name
    if script == 'compare_generated_distributions.py':
        output = Path(command[command.index('--output') + 1])
        split = command[command.index('--reference-split') + 1]
        model = command[command.index('--model') + 1]
        write_json(output, {'metadata': {'reference_split': None if split == 'full' else split,
                                        'model': model,
                                        'representations': [source.split('=', 1)[0] for index, source in enumerate(command)
                                                            if index and command[index - 1] == '--generated']},
                            'reference': {'num_input_graphs': 10, 'num_valid_molecules': 10}})
        return
    output = Path(command[command.index('--output-dir') + 1])
    output.mkdir(parents=True, exist_ok=True)
    if script == 'compare_three_representations.py':
        (output / 'overview.png').write_bytes(b'figure fixture')
    elif script == 'score_generated_distributions.py':
        report = json.loads(Path(command[command.index('--report') + 1]).read_text())
        quality = {'num_input_graphs': 10, 'valid_fraction': .9, 'uniqueness_fraction': .8,
                   'mmff_coverage': {'valid_molecule_fraction': 1.}}
        write_json(output / 'metrics.json', {'metadata': report['metadata'], 'representations': {
            name: {'available_seeds': [0, 1, 2], 'runs': [{'quality': quality}] * 3}
            for name in report['metadata']['representations']}})
        (output / 'summary.md').write_text('# Distanze\n\n[csv](wasserstein_summary.csv)\n'
                                           '![Wasserstein](wasserstein.png)\n'
                                           '[SciPy](https://docs.scipy.org/)\n\n## Dettagli\n')
        (output / 'wasserstein_summary.csv').write_text('descriptor,value\n')
        for name in ('ranking.png', 'groups_tvd.png', 'wasserstein.png'):
            (output / name).write_bytes(b'figure fixture')
    elif script == 'compare_distribution_statistics.py':
        label = command[command.index('--scores') + 1].split('=', 1)[0]
        write_json(output / 'statistics.json', {'contexts': {label: {}}})
        (output / 'summary.md').write_text('# Test statistici\n\np Holm: 1.0\n')
    else:
        raise AssertionError(script)


class DistributionWorkflowTests(unittest.TestCase):
    def test_explicit_wandb_roots_are_forwarded_to_analysis(self):
        args = workflow.parse_args(['--wandb-root', '/remote/logs', '--wandb-root', '/remote/archive'])
        self.assertEqual(args.wandb_root, [Path('/remote/logs'), Path('/remote/archive')])
        command = workflow.scope_commands(args, 'd4', 'test', Path('/output'), ['base=/samples.pkl'])[0][1]
        roots = [command[index + 1] for index, token in enumerate(command) if token == '--wandb-root']
        self.assertEqual(roots, ['/remote/logs', '/remote/archive'])

    def test_source_selection_failure_preserves_discovery_diagnostics(self):
        from types import SimpleNamespace
        import sys
        def missing(paths, checkpoints, seeds, messages, roots):
            messages.append('training seed is unknown: offline metadata missing')
            return []
        analysis = SimpleNamespace(resolve_dataset_name=lambda name: name,
                                   experiment_paths=lambda *args: {'base': [Path('/samples')]},
                                   discover_runs=missing)
        with patch.dict(sys.modules, {'compare_generated_distributions': analysis}):
            with self.assertRaisesRegex(ValueError, 'offline metadata missing'):
                workflow.select_sources(workflow.parse_args(['--reference-dataset', 'gdb13']), 'd4')

    def test_default_plan_and_explicit_aliases_keep_splits_separate(self):
        args = workflow.parse_args([])
        self.assertEqual(args.splits, ['train', 'validation', 'test'])
        self.assertEqual(args.models, ['d4', 'midi'])
        args = workflow.parse_args(['--reference-splits', 'valid', 'val', 'test', '--model', 'midi'])
        self.assertEqual(args.splits, ['validation', 'test'])
        self.assertEqual(args.models, ['midi'])
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            workflow.parse_args(['--reference-dataset', '../qm9'])

    def test_default_workflow_writes_six_clean_reports_and_pins_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def sources(args, model):
                return [f'base=/saved/{model}/seed0.pkl'], []
            with patch.object(workflow, 'select_sources', side_effect=sources) as select, \
                    patch.object(workflow.subprocess, 'run', side_effect=fake_step) as run, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(workflow.main(['--output-root', directory]), 0)
            self.assertEqual(select.call_count, 2)
            self.assertEqual(run.call_count, 24)
            for model in ['d4', 'midi']:
                for split in ['train', 'validation', 'test']:
                    folder = root / 'qm9' / model / split
                    self.assertEqual({path.name for path in folder.iterdir()}, {'REPORT.md', 'images', 'details'})
                    self.assertEqual({path.name for path in (folder / 'images').iterdir()},
                                     {'overview.png', 'ranking.png', 'groups_tvd.png'})
                    text = (folder / 'REPORT.md').read_text()
                    self.assertIn('Distanze', text)
                    self.assertIn('Test statistici', text)
                    self.assertIn('details/scores/wasserstein_summary.csv', text)
                    self.assertIn('![Wasserstein](details/scores/wasserstein.png)', text)
                    self.assertIn('[SciPy](https://docs.scipy.org/)', text)
                    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
                        if not target.startswith('https://'):
                            self.assertTrue((folder / target).is_file(), target)
                    stats = json.loads((folder / 'details/statistics/statistics.json').read_text())
                    self.assertEqual(set(stats['contexts']), {split})
            analysis_calls = [call.args[0] for call in run.call_args_list
                              if Path(call.args[0][1]).name == 'compare_generated_distributions.py']
            self.assertEqual([command[command.index('--reference-split') + 1] for command in analysis_calls],
                             ['train', 'valid', 'test'] * 2)
            for command in analysis_calls:
                model = command[command.index('--model') + 1]
                self.assertEqual(command[command.index('--generated') + 1], f'base=/saved/{model}/seed0.pkl')
                self.assertIn('--skip-plots', command)

    def test_single_split_uses_dataset_and_model_in_paths(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(workflow, 'select_sources', return_value=(['base=/saved/input.pkl'], [])), \
                patch.object(workflow.subprocess, 'run', side_effect=fake_step) as run, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(workflow.main(['--reference-dataset', 'gdb13', '--model', 'midi',
                                           '--reference-split', 'test', '--output-root', directory]), 0)
            self.assertEqual(run.call_count, 4)
            self.assertTrue((Path(directory) / 'gdb13/midi/test/REPORT.md').is_file())
            self.assertFalse((Path(directory) / 'gdb13/midi/train').exists())

    def test_two_representation_report_names_only_the_compared_variants(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(workflow, 'select_sources', return_value=(['base=/base.pkl', 'atom_details=/details.pkl'], [])), \
                patch.object(workflow.subprocess, 'run', side_effect=fake_step), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(workflow.main(['--reference-dataset', 'zinc', '--model', 'd4',
                                           '--reference-split', 'test', '--output-root', directory]), 0)
            report = (Path(directory) / 'zinc/d4/test/REPORT.md').read_text()
            self.assertIn('Base + Atom details', report)
            self.assertNotIn('Base + MMFF', report)
            self.assertNotIn('| MMFF |', report)
            self.assertNotIn('tre rappresentazioni', report)

    def test_failed_split_replaces_stale_report_and_other_splits_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / 'qm9/midi/train/REPORT.md'
            old.parent.mkdir(parents=True)
            old.write_text('OLD SUCCESS')
            def fail_train(command, **kwargs):
                if '--reference-split' in command and command[command.index('--reference-split') + 1] == 'train':
                    raise subprocess.CalledProcessError(1, command)
                return fake_step(command, **kwargs)
            with patch.object(workflow, 'select_sources', return_value=(['base=/saved/input.pkl'], [])), \
                    patch.object(workflow.subprocess, 'run', side_effect=fail_train), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(workflow.main(['--model', 'midi', '--output-root', directory]), 1)
            self.assertIn('Analisi non completata', old.read_text())
            self.assertNotIn('OLD SUCCESS', old.read_text())
            self.assertIn('Test statistici', (old.parent.parent / 'validation/REPORT.md').read_text())
            self.assertIn('Test statistici', (old.parent.parent / 'test/REPORT.md').read_text())

    def test_missing_generation_sources_create_explicit_failure_reports(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(workflow, 'select_sources', side_effect=ValueError('No generated files')), \
                patch.object(workflow.subprocess, 'run') as run, \
                redirect_stderr(io.StringIO()):
            self.assertEqual(workflow.main(['--model', 'midi', '--output-root', directory]), 1)
            run.assert_not_called()
            for split in workflow.DEFAULT_SPLITS:
                text = (Path(directory) / 'qm9/midi' / split / 'REPORT.md').read_text()
                self.assertIn('No generated files', text)


if __name__ == '__main__':
    unittest.main()
