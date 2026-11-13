"""Find local training-seed evidence without loading molecules or checkpoints."""

import json
import os
from pathlib import Path
import re


def metadata_search_roots(run_dir, checkpoints_root, wandb_roots=()):
    run_dir, checkpoints_root = Path(run_dir), Path(checkpoints_root)
    roots = [checkpoints_root / "wandb", checkpoints_root.parent / "wandb",
             run_dir.parent.parent / "wandb", run_dir.parent.parent.parent / "wandb",
             Path.cwd() / "wandb", Path(__file__).resolve().parent / "wandb"]
    configured = list(wandb_roots)
    if os.environ.get("WANDB_DIR"):
        configured.append(os.environ["WANDB_DIR"])
    for root in configured:
        root = Path(root).expanduser()
        roots.extend((root, root / "wandb"))
    return sorted({root.resolve() for root in roots})


def unwrap_config(value):
    return value.get("value") if isinstance(value, dict) and "value" in value else value


def wandb_metadata_directories(run_dir, checkpoints_root, wandb_roots=()):
    """Match stable IDs even if the local vN prefix was renamed after training."""
    run_dir = Path(run_dir)
    identities = {run_dir.name}
    match = re.fullmatch(r"v\d+_(.+)", run_dir.name)
    if match:
        identities.add(match.group(1))
    files_dirs = set()
    for root in metadata_search_roots(run_dir, checkpoints_root, wandb_roots):
        if not root.is_dir():
            continue
        for pattern in ("run-*", "offline-run-*"):
            for directory in root.glob(pattern):
                saved_name = directory.name.rsplit("-", 1)[-1]
                saved_id = re.sub(r"^v\d+_", "", saved_name)
                if (any(directory.name.endswith("-" + identity) for identity in identities)
                        or saved_id in identities):
                    files_dirs.add((directory / "files").resolve())
    return files_dirs


def read_run_metadata(run_dir, checkpoints_root, wandb_roots=()):
    """Read saved seed evidence; a vN counter never supplies a training seed.

    W&B may use either the full local run name or its ID, and may be offline.
    Version prefixes can differ after local renaming; the stable ID must match.
    All matching evidence is checked, including conflicting resumed-run seeds.
    """
    run_dir = Path(run_dir)
    files_dirs = wandb_metadata_directories(run_dir, checkpoints_root, wandb_roots)

    configs = {run_dir / "config.yaml", run_dir / ".hydra/config.yaml"}
    configs.update(directory / "config.yaml" for directory in files_dirs)
    evidence, dataset_names = [], set()
    for path in sorted(configs):
        if not path.is_file():
            continue
        import yaml
        config = yaml.safe_load(path.read_text()) or {}
        value = unwrap_config(config.get("seed"))
        if value is not None:
            evidence.append((int(value), str(path)))
        task = unwrap_config(config.get("task", {})) or {}
        dataset = unwrap_config(task.get("dataset", {})) or {}
        if dataset.get("name"):
            dataset_names.add(unwrap_config(dataset["name"]))
    for directory in sorted(files_dirs):
        path = directory / "wandb-metadata.json"
        if not path.is_file():
            continue
        for arg in json.loads(path.read_text()).get("args", []):
            match = re.fullmatch(r"\+{0,2}seed=(-?\d+)", arg)
            if match:
                evidence.append((int(match.group(1)), str(path)))
    seeds = {seed for seed, _ in evidence}
    if len(seeds) > 1 or len(dataset_names) > 1:
        raise ValueError(f"Conflicting run metadata in {run_dir}: {evidence}, datasets={dataset_names}")
    return {"seed": next(iter(seeds), None),
            "seed_sources": sorted({path for _, path in evidence}),
            "dataset_name": next(iter(dataset_names), None)}
