"""Local seed discovery, without chemistry, training, or W&B network access."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from distribution_run_metadata import read_run_metadata


class RunMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoints = self.root / "project/checkpoints"
        self.run = self.checkpoints / "gdb13-d4/v0_glejzg5c"
        self.run.mkdir(parents=True)
        self.environment = patch.dict(os.environ, {"WANDB_DIR": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def metadata(self, root, name, seed):
        files = root / name / "files"
        files.mkdir(parents=True)
        path = files / "wandb-metadata.json"
        path.write_text(json.dumps({"args": ["+preset/final/d4/gdb13=base", f"seed={seed}"]}))
        return path

    def test_online_full_run_name(self):
        path = self.metadata(self.checkpoints / "wandb", "run-20261102-v0_glejzg5c", 2)
        result = read_run_metadata(self.run, self.checkpoints)
        self.assertEqual(result["seed"], 2)
        self.assertEqual(result["seed_sources"], [str(path)])

    def test_online_bare_wandb_id(self):
        self.metadata(self.checkpoints / "wandb", "run-20261102-glejzg5c", 1)
        self.assertEqual(read_run_metadata(self.run, self.checkpoints)["seed"], 1)

    def test_offline_both_identity_formats(self):
        for identity in ("glejzg5c", "v0_glejzg5c"):
            with self.subTest(identity=identity):
                self.metadata(self.checkpoints / "wandb", f"offline-run-20261102-{identity}", 2)
                self.assertEqual(read_run_metadata(self.run, self.checkpoints)["seed"], 2)

    def test_repository_wandb_root(self):
        self.metadata(self.checkpoints.parent / "wandb", "run-20261102-glejzg5c", 1)
        self.assertEqual(read_run_metadata(self.run, self.checkpoints)["seed"], 1)

    def test_explicit_external_root(self):
        external = self.root / "home/logs/wandb"
        self.metadata(external, "offline-run-20261102-glejzg5c", 2)
        self.assertIsNone(read_run_metadata(self.run, self.checkpoints)["seed"])
        self.assertEqual(read_run_metadata(self.run, self.checkpoints, [external])["seed"], 2)

    def test_environment_root_with_wandb_child(self):
        external = self.root / "logs"
        self.metadata(external / "wandb", "offline-run-20261102-glejzg5c", 1)
        with patch.dict(os.environ, {"WANDB_DIR": str(external)}):
            self.assertEqual(read_run_metadata(self.run, self.checkpoints)["seed"], 1)

    def test_conflicting_online_offline_metadata_fail(self):
        self.metadata(self.checkpoints / "wandb", "run-20261102-v0_glejzg5c", 0)
        self.metadata(self.checkpoints / "wandb", "offline-run-20261102-glejzg5c", 1)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            read_run_metadata(self.run, self.checkpoints)

    def test_unknown_seed_and_similar_unrelated_ids_are_not_guessed(self):
        self.metadata(self.checkpoints / "wandb", "run-20261102-otherglejzg5c", 1)
        result = read_run_metadata(self.run, self.checkpoints)
        self.assertIsNone(result["seed"])
        self.assertEqual(result["seed_sources"], [])

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML is not installed")
    def test_saved_wandb_yaml_supplies_seed_and_dataset(self):
        files = self.checkpoints / "wandb/offline-run-20261102-glejzg5c/files"
        files.mkdir(parents=True)
        (files / "config.yaml").write_text("seed:\n  value: 2\ntask:\n  value:\n    dataset:\n      name: gdb13\n")
        result = read_run_metadata(self.run, self.checkpoints)
        self.assertEqual((result["seed"], result["dataset_name"]), (2, "gdb13"))


if __name__ == "__main__":
    unittest.main()
