#!/usr/bin/env python3
"""Compare saved generations against train, validation and test separately.

Default output: outputs/distribution_comparison/<dataset>/<model>/<split>/REPORT.md.
Only selected report images share the public output; intermediate artifacts live
under details/. No training, generation or dataset preprocessing is performed.
"""

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parent
DEFAULT_SPLITS = ("train", "validation", "test")
SPLIT_ALIASES = {"training": "train", "train": "train", "validation": "validation",
                 "valid": "validation", "val": "validation", "test": "test", "full": "full"}
LABELS = {"base": "Base", "mmff": "MMFF", "atom_details": "Atom details"}
MODEL_LABELS = {"d4": "D4", "midi": "MiDi"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dataset", default="qm9")
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--reference-split", "--reference-splits", nargs="+", dest="splits",
                        choices=tuple(SPLIT_ALIASES), default=list(DEFAULT_SPLITS),
                        help="Reference splits (default: train validation test); valid/val are aliases for validation")
    parser.add_argument("--model", "--models", nargs="+", dest="models", choices=tuple(MODEL_LABELS),
                        default=list(MODEL_LABELS))
    parser.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--wandb-root", type=Path, action="append", default=[],
                        help="Additional local W&B directory (repeatable; online and offline runs)")
    parser.add_argument("--representations", nargs="+", choices=tuple(LABELS), default=list(LABELS))
    parser.add_argument("--generated", action="append", help="Explicit NAME=PATH source; requires one --model")
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-reference", type=int)
    parser.add_argument("--max-generated", type=int)
    parser.add_argument("--seed", type=int, default=0, help="Subsampling/permutation RNG seed, not a training seed")
    parser.add_argument("--top-k", "--plot-top-k", type=int, default=20)
    parser.add_argument("--no-pdf", action="store_true", help="Omit PDFs from detailed distribution plots")
    parser.add_argument("--group-weights", nargs="+", default=[])
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--max-permutations", type=int, default=20000)
    parser.add_argument("--output-root", "--output-dir", type=Path, default=Path("outputs/distribution_comparison"))
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.reference_dataset):
        parser.error("--reference-dataset must be a dataset name, not a path; use --reference-root for paths")
    args.models = list(dict.fromkeys(args.models))
    args.splits = list(dict.fromkeys(SPLIT_ALIASES[split] for split in args.splits))
    args.representations = list(dict.fromkeys(args.representations))
    if args.generated and len(args.models) != 1:
        parser.error("Explicit --generated inputs require exactly one --model")
    if len(set(args.expected_seeds)) != len(args.expected_seeds):
        parser.error("--expected-seeds must be distinct")
    for name in ("max_reference", "max_generated", "top_k", "max_permutations"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error("--" + name.replace("_", "-") + " must be positive")
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1")
    args.wandb_root = [root.expanduser().resolve() for root in args.wandb_root]
    args.output_root = args.output_root.resolve()
    args.checkpoints_root = args.checkpoints_root.resolve()
    if args.reference_root is not None:
        args.reference_root = args.reference_root.resolve()
    return args


def select_sources(args, model):
    """Pin generation files once, so all reference splits use the same samples."""
    import compare_generated_distributions as analysis

    dataset = analysis.resolve_dataset_name(args.reference_dataset)
    family = re.sub(r"_(mmff|atom_details)$", "", dataset)
    messages, selected = [], []
    # experiment_paths reads args.model; the orchestrator supports several models.
    options = argparse.Namespace(**vars(args), model=model)
    for name, paths in analysis.experiment_paths(options, family, messages).items():
        for run in analysis.discover_runs(paths, args.checkpoints_root, args.expected_seeds, messages, args.wandb_root):
            selected.append(f"{name}={run['generated_file']}")
    if not selected:
        diagnostics = "\n".join(messages)
        raise ValueError(f"No saved generations with identifiable training seeds for {dataset}/{model}"
                         + ("\n\nDiscovery diagnostics:\n" + diagnostics if diagnostics else ""))
    return selected, messages


def scope_commands(args, model, split, directory, sources):
    details = directory / "details"
    raw = details / "analysis/report.json"
    cache_split = "valid" if split == "validation" else split
    analysis = [sys.executable, str(REPOSITORY / "compare_generated_distributions.py"),
                "--reference-dataset", args.reference_dataset, "--reference-split", cache_split,
                "--model", model, "--checkpoints-root", str(args.checkpoints_root),
                "--representations", *args.representations,
                "--expected-seeds", *map(str, args.expected_seeds), "--seed", str(args.seed),
                "--output", str(raw), "--skip-plots"]
    for root in args.wandb_root:
        analysis += ["--wandb-root", str(root)]
    for source in sources:
        analysis += ["--generated", source]
    for option in ("reference_root", "max_reference", "max_generated"):
        if getattr(args, option) is not None:
            analysis += ["--" + option.replace("_", "-"), str(getattr(args, option))]
    plots = [sys.executable, str(REPOSITORY / "compare_three_representations.py"),
             "--report", str(raw), "--output-dir", str(details / "distributions"), "--top-k", str(args.top_k)]
    if args.no_pdf:
        plots.append("--no-pdf")
    scores = [sys.executable, str(REPOSITORY / "score_generated_distributions.py"),
              "--report", str(raw), "--output-dir", str(details / "scores")]
    if args.group_weights:
        scores += ["--group-weights", *args.group_weights]
    inference = [sys.executable, str(REPOSITORY / "compare_distribution_statistics.py"),
                 "--scores", f"{split}={details / 'scores/metrics.json'}",
                 "--output-dir", str(details / "statistics"), "--alpha", str(args.alpha),
                 "--max-permutations", str(args.max_permutations), "--seed", str(args.seed)]
    return [("analysis", analysis), ("plots", plots), ("scores", scores), ("statistics", inference)]


def report_section(text, prefix):
    """Demote section headings and rebase local Markdown links into details/."""
    text = re.sub(r"(?m)^(#{1,5}) ", r"#\1 ", text)

    def replace(match):
        target = match.group(2)
        if target.startswith(("http://", "https://", "#", "/", "mailto:")):
            return match.group(0)
        return match.group(1) + prefix + "/" + target + match.group(3)

    return re.sub(r"(!?\[[^\]]*\]\()([^\s)]+)(\))", replace, text)


def percent(values):
    values = [value for value in values if value is not None]
    return f"{100 * statistics.mean(values):.3f}%" if values else "n/d"


def write_report(directory, dataset, model, split, messages=()):
    details = directory / "details"
    analysis = json.loads((details / "analysis/report.json").read_text())
    scores = json.loads((details / "scores/metrics.json").read_text())
    inference = json.loads((details / "statistics/statistics.json").read_text())
    expected = "valid" if split == "validation" else None if split == "full" else split
    for document in (analysis, scores):
        if document["metadata"]["reference_split"] != expected or document["metadata"]["model"] != model:
            raise ValueError("Intermediate report does not match the requested model/reference split")
    if set(inference["contexts"]) != {split}:
        raise ValueError("Statistical report must contain only the current reference split")

    images = directory / "images"
    images.mkdir(exist_ok=True)
    selected_images = []
    representations = " + ".join(LABELS[name] for name in scores["representations"])
    for source, filename, caption in (
        (details / "distributions/overview.png", "overview.png", f"Distribuzioni: dataset + {representations}"),
        (details / "scores/ranking.png", "ranking.png", "Distanze complessive per seed"),
        (details / "scores/groups_tvd.png", "groups_tvd.png", "Distanze per famiglia di descrittori"),
    ):
        if source.is_file():
            shutil.copy2(source, images / filename)
            selected_images.append(f"![{caption}](images/{filename})")

    reference = analysis["reference"]
    lines = [f"# Distribuzioni molecolari — {dataset} · {MODEL_LABELS[model]} · {split}", "",
             f"Riferimento: **{split}**; grafi analizzati: **{reference['num_input_graphs']}**; "
             f"molecole valide: **{reference['num_valid_molecules']}**.", "",
             f"Questo report confronta {representations} per un solo modello e un solo split. "
             "Gli altri split sono valutati separatamente, usando gli stessi file di generazione.", "",
             "## Generazioni analizzate", "",
             "| Rappresentazione | Seed | Grafi analizzati | Validità media | Unicità media | Copertura MMFF media |",
             "|---|---|---:|---:|---:|---:|"]
    for name, entry in scores["representations"].items():
        quality = [run["quality"] for run in entry["runs"]]
        count = sum(item.get("num_input_graphs") or 0 for item in quality)
        coverage = [(item.get("mmff_coverage") or {}).get("valid_molecule_fraction") for item in quality]
        lines.append(f"| {LABELS[name]} | {', '.join(map(str, entry['available_seeds'])) or 'nessuno'} | "
                     f"{count} | {percent([item.get('valid_fraction') for item in quality])} | "
                     f"{percent([item.get('uniqueness_fraction') for item in quality])} | {percent(coverage)} |")
    if selected_images:
        lines += ["", "## Grafici principali", "", "\n\n".join(selected_images)]
    lines += ["", report_section((details / "scores/summary.md").read_text(), "details/scores"),
              "", report_section((details / "statistics/summary.md").read_text(), "details/statistics")]
    messages = list(dict.fromkeys(messages))
    if messages:
        lines += ["", "## Selezione delle generazioni", "", *[f"- {message}" for message in messages]]
    lines += ["", "## Dati di dettaglio", "",
              "I dati intermedi e i grafici completi sono conservati in `details/`.", "",
              "- [Conteggi, descrittori e file di generazione](details/analysis/report.json)",
              "- [Punteggi per seed e definizione dell’indice](details/scores/metrics.json)",
              "- [Test statistici completi](details/statistics/statistics.json)", ""]
    temporary = details / "REPORT.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(directory / "REPORT.md")


def write_failure(directory, dataset, model, split, stage, error):
    (directory / "details").mkdir(parents=True, exist_ok=True)
    (directory / "REPORT.md").write_text(
        f"# Analisi non completata — {dataset}/{model}/{split}\n\n"
        f"Passaggio fallito: `{stage}`.\n\n{error}\n\n"
        "Gli eventuali file intermedi in `details/` non costituiscono un risultato completo.\n",
        encoding="utf-8",
    )


def run_scope(args, model, split, sources, messages):
    directory = args.output_root / args.reference_dataset / model / split
    (directory / "details").mkdir(parents=True, exist_ok=True)
    report = directory / "REPORT.md"
    report.write_text(f"# Analisi in corso — {args.reference_dataset}/{model}/{split}\n\n"
                      "Il report sarà pubblicato al completamento di tutti i passaggi.\n", encoding="utf-8")
    stage = "preparation"
    try:
        for stage, command in scope_commands(args, model, split, directory, sources):
            print(f"[{args.reference_dataset}/{model}/{split}] {stage}", flush=True)
            subprocess.run(command, check=True, cwd=REPOSITORY)
        stage = "report"
        write_report(directory, args.reference_dataset, model, split, messages)
    except Exception as error:
        write_failure(directory, args.reference_dataset, model, split, stage, error)
        raise
    print(f"Report: {report}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    failed = False
    for model in args.models:
        try:
            sources, messages = select_sources(args, model)
        except Exception as error:
            print(f"{args.reference_dataset}/{model}: {error}", file=sys.stderr)
            for split in args.splits:
                directory = args.output_root / args.reference_dataset / model / split
                write_failure(directory, args.reference_dataset, model, split, "source selection", error)
            failed = True
            continue
        for split in args.splits:
            try:
                run_scope(args, model, split, sources, messages)
            except Exception as error:
                print(f"{args.reference_dataset}/{model}/{split}: {error}", file=sys.stderr)
                failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
