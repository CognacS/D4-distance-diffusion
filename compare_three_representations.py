#!/usr/bin/env python3
"""Plot dataset + base + MMFF + atom_details together on identical axes.

Reuse an existing analysis, without loading molecule pickles again:
    python compare_three_representations.py --report PATH/TO/report.json

Or compute a new analysis using the existing analyzer's dataset/seed options:
    python compare_three_representations.py --reference-dataset qm9 --reference-split test
    python compare_three_representations.py --reference-dataset gdb13 --reference-split valid

The original analysis script and its existing reports/plots are left unchanged.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import compare_generated_distributions as analysis


REPRESENTATIONS = ("base", "mmff", "atom_details")
LABELS = {"dataset": "Dataset", "base": "Base", "mmff": "MMFF", "atom_details": "Atom details"}
COLORS = {"dataset": "#929292", "base": "#4477AA", "mmff": "#EE7733", "atom_details": "#228833"}
TITLES = {
    "atom_symbols": "Elementi", "formal_charges": "Cariche formali degli atomi",
    "molecular_charge": "Carica totale della molecola", "node_mmff_type": "Tipi atomici MMFF",
    "node_is_in_ring": "Atomi appartenenti ad anelli", "node_is_aromatic": "Aromaticità degli atomi",
    "node_hybridization": "Ibridazione degli atomi", "bond_types": "Tipi di legame",
    "ring_sizes": "Dimensioni degli anelli", "substructure_presence": "Molecole con gruppi funzionali",
    "element_charge": "Elemento e carica formale", "mmff_charge": "Tipo MMFF e carica formale",
}
OVERVIEW_FIELDS = ("node_mmff_type", "formal_charges", "node_hybridization",
                   "node_is_in_ring", "node_is_aromatic", "bond_types")


@contextmanager
def command_arguments(arguments):
    previous = sys.argv
    sys.argv = [str(Path(analysis.__file__)), *arguments]
    try:
        yield
    finally:
        sys.argv = previous


def distributions(aggregate):
    """Use exact saved counts when available, rather than rounded JSON probabilities."""
    result = analysis.flat_distributions(aggregate)
    for field, counts in aggregate.get("counts", {}).items():
        result[field] = analysis.normalize_counter(Counter(counts))
    for field, counts in aggregate.get("dataset_specific_atom_feature_counts", {}).items():
        result[field] = analysis.normalize_counter(Counter(counts))
    return result


def defined(record):
    return bool(record) and any(value is not None for value in record.values())


def collect_distributions(report):
    if report.get("schema_version") != 2 or "representations" not in report:
        raise ValueError("Serve il report multi-seed (schema_version=2) di compare_generated_distributions.py")
    reference = distributions(report["reference"])
    groups = {
        name: [{"seed": run["seed"], "distributions": distributions(run["generated"])}
               for run in report["representations"].get(name, {}).get("runs", [])]
        for name in REPRESENTATIONS
    }
    if not any(groups.values()):
        raise ValueError("Il report non contiene generazioni utilizzabili")
    return reference, groups


def comparison_table(field, reference, groups, top_k):
    """One common support and category order for all four plotted populations."""
    ref = reference.get(field, {})
    selected_runs = {
        name: [run for run in groups[name] if defined(run["distributions"].get(field))]
        for name in REPRESENTATIONS
    }
    supports = [ref] + [run["distributions"][field] for runs in selected_runs.values() for run in runs]
    support = {key for record in supports for key, value in record.items() if value is not None}
    scores = {
        key: (ref.get(key) or 0) + sum(
            statistics.mean(run["distributions"][field].get(key) or 0 for run in runs)
            for runs in selected_runs.values() if runs)
        for key in support
    }
    labels = sorted(support, key=lambda key: (-scores[key], key))[:top_k]
    omitted = support - set(labels)
    # Presence rates describe overlapping groups, not mutually exclusive categories.
    add_other = bool(omitted) and field != "substructure_presence"
    rows = []
    for category in labels + (["Altre categorie"] if add_other else []):
        keys = omitted if add_other and category == "Altre categorie" else {category}
        ref_value = sum(ref.get(key) or 0 for key in keys) if defined(ref) else None
        row = {"category": category, "dataset": ref_value}
        for name, runs in selected_runs.items():
            values = [sum(run["distributions"][field].get(key) or 0 for key in keys) for run in runs]
            stat = analysis.numeric_summary(values)
            row[name] = {**stat, "seeds": [run["seed"] for run in runs],
                         "difference_percentage_points": 100 * (stat["mean"] - ref_value)
                         if stat["mean"] is not None and ref_value is not None else None}
        rows.append(row)
    return rows


def legend_handles(rows):
    reference_label = "Dataset" if rows and rows[0]["dataset"] is not None else "Dataset (non disponibile)"
    handles = [Patch(facecolor=COLORS["dataset"], label=reference_label)]
    for name in REPRESENTATIONS:
        seeds = rows[0][name]["seeds"] if rows else []
        suffix = f"seed {', '.join(map(str, seeds))}" if seeds else "non disponibile"
        handles.append(Patch(facecolor=COLORS[name], label=f"{LABELS[name]} ({suffix})"))
    return handles


def draw_grouped(ax, rows, field, show_labels=True):
    positions = list(range(len(rows)))
    width = 0.19
    ref = [(index - 1.5 * width, row["dataset"]) for index, row in enumerate(rows) if row["dataset"] is not None]
    if ref:
        ax.bar([p for p, _ in ref], [100 * value for _, value in ref], width, color=COLORS["dataset"])
    for offset, name in enumerate(REPRESENTATIONS, 1):
        available = [(index, row[name]) for index, row in enumerate(rows) if row[name]["mean"] is not None]
        if available:
            ax.bar([index + (offset - 1.5) * width for index, _ in available],
                   [100 * stat["mean"] for _, stat in available], width, color=COLORS[name])
            errorbars = [(index, stat) for index, stat in available if stat["std"] is not None]
            if errorbars:
                ax.errorbar([index + (offset - 1.5) * width for index, _ in errorbars],
                            [100 * stat["mean"] for _, stat in errorbars],
                            yerr=[100 * stat["std"] for _, stat in errorbars],
                            fmt="none", ecolor="#252525", capsize=2, linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([row["category"] for row in rows] if show_labels else [], rotation=45, ha="right")
    ax.set_ylabel("Molecole valide (%)" if field in {"molecular_charge", "substructure_presence"} else "Frequenza (%)")
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot_descriptor(field, rows, title):
    figure, (top, difference) = plt.subplots(
        2, 1, figsize=(max(10, min(23, len(rows) * 0.9)), 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    draw_grouped(top, rows, field, show_labels=False)
    top.set_title(title + "\nDataset + Base + MMFF + Atom details", pad=12)
    top.legend(handles=legend_handles(rows), loc="upper right", fontsize=9)
    for offset, name in enumerate(REPRESENTATIONS):
        available = [(index, row[name]) for index, row in enumerate(rows)
                     if row[name]["difference_percentage_points"] is not None]
        if available:
            difference.bar([index + (offset - 1) * 0.24 for index, _ in available],
                           [stat["difference_percentage_points"] for _, stat in available],
                           0.24, color=COLORS[name])
            errorbars = [(index, stat) for index, stat in available if stat["std"] is not None]
            if errorbars:
                difference.errorbar([index + (offset - 1) * 0.24 for index, _ in errorbars],
                                    [stat["difference_percentage_points"] for _, stat in errorbars],
                                    yerr=[100 * stat["std"] for _, stat in errorbars],
                                    fmt="none", ecolor="#252525", capsize=2, linewidth=0.8)
    difference.axhline(0, color="#454545", linewidth=1)
    difference.set_ylabel("Scarto dal dataset\n(punti percentuali)")
    difference.set_xticks(range(len(rows)))
    difference.set_xticklabels([row["category"] for row in rows], rotation=45, ha="right")
    difference.grid(axis="y", alpha=0.18)
    difference.set_axisbelow(True)
    difference.spines[["top", "right"]].set_visible(False)
    figure.supxlabel("Media tra seed disponibili; barre di errore = deviazione standard campionaria (n−1)", fontsize=9)
    return figure


def save_figure(figure, stem, pdf):
    paths = [stem.with_suffix(".png")]
    figure.savefig(paths[0], dpi=170)
    if pdf:
        paths.append(stem.with_suffix(".pdf"))
        figure.savefig(paths[-1])
    plt.close(figure)
    return [str(path) for path in paths]


def make_plots(report, output_dir, top_k=20, pdf=True):
    reference, groups = collect_distributions(report)
    warnings = list(report.get("warnings", []))
    expected = set(report["metadata"].get("expected_seeds", [0, 1, 2]))
    for name, runs in groups.items():
        seeds = {run["seed"] for run in runs}
        missing = sorted(expected - seeds)
        if missing:
            warnings.append(f"{LABELS[name]}: seed disponibili {sorted(seeds)}, mancanti {missing}")
    fields = sorted(set(reference) | {field for runs in groups.values() for run in runs for field in run["distributions"]})
    tables, paths = {}, []
    output_dir.mkdir(parents=True, exist_ok=True)
    for field in fields:
        rows = comparison_table(field, reference, groups, top_k)
        if not rows:
            continue
        tables[field] = rows
        if rows[0]["dataset"] is None:
            warnings.append(f"{field}: distribuzione del dataset non disponibile")
        missing_fields = [LABELS[name] for name in REPRESENTATIONS if not rows[0][name]["n"]]
        if missing_fields:
            warnings.append(f"{field}: distribuzione non disponibile per {', '.join(missing_fields)}")
        figure = plot_descriptor(field, rows, TITLES.get(field, field.replace("_", " ")))
        paths.extend(save_figure(figure, output_dir / analysis.sanitize_plot_filename(field), pdf))

    overview_fields = [field for field in OVERVIEW_FIELDS if field in tables]
    if overview_fields:
        figure, axes = plt.subplots(3, 2, figsize=(19, 15), layout="constrained")
        for ax, field in zip(axes.flat, overview_fields):
            rows = comparison_table(field, reference, groups, min(top_k, 10))
            draw_grouped(ax, rows, field)
            ax.set_title(TITLES[field])
            ax.tick_params(axis="x", labelsize=8)
            ax.legend(handles=legend_handles(rows), fontsize=7)
        for ax in list(axes.flat)[len(overview_fields):]:
            ax.set_visible(False)
        metadata = report["metadata"]
        figure.suptitle(f"{metadata['reference_dataset']} — {metadata['reference_split'] or 'full'}\n"
                       "Dataset + Base + MMFF + Atom details · media e deviazione standard tra seed", fontsize=17)
        paths.extend(save_figure(figure, output_dir / "overview", pdf))

    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", newline="") as stream:
        columns = ["descriptor", "category", "dataset"] + [
            f"{name}_{key}" for name in REPRESENTATIONS
            for key in ["mean", "std", "n", "difference_percentage_points"]]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for field, rows in tables.items():
            for row in rows:
                writer.writerow({"descriptor": field, "category": row["category"], "dataset": row["dataset"],
                                 **{f"{name}_{key}": row[name][key] for name in REPRESENTATIONS
                                    for key in ["mean", "std", "n", "difference_percentage_points"]}})
    for message in warnings:
        print("WARNING:", message, file=sys.stderr)
    return {"metadata": report["metadata"], "warnings": warnings, "tables": tables,
            "plot_files": paths, "csv_file": str(csv_path),
            "notes": ["All four populations share category order, support and axes.",
                      "Frequencies are fractions in JSON/CSV and percentages in figures; differences are percentage points.",
                      "Seed means have equal weights; std uses n-1 and is undefined for one seed.",
                      "Missing distributions are undefined, never replaced by zero-probability populations.",
                      "Tables/CSV describe the plotted categories, including grouped Other categories; "
                      "the source report retains all individual categories."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="Senza --report sono accettate tutte le opzioni del precedente script, "
                                            "tra cui --generated NAME=PATH, --model, --reference-root e --max-generated.")
    parser.add_argument("--report", type=Path, help="Report multi-seed già calcolato")
    parser.add_argument("--output-dir", type=Path, help="Nuova cartella per grafici e tabelle")
    parser.add_argument("--top-k", type=int, default=20, help="Categorie per grafico (default 20)")
    parser.add_argument("--no-pdf", action="store_true", help="Produci solo PNG")
    options, analysis_options = parser.parse_known_args()
    if options.top_k < 1:
        parser.error("--top-k deve essere positivo")
    if options.report and analysis_options:
        parser.error("Con --report dataset, split e seed vengono dal report; omettere le opzioni di analisi")
    if not options.report:
        with command_arguments(analysis_options):
            original_options = analysis.parse_args()
        dataset, split = original_options.reference_dataset, analysis.normalize_split(original_options.reference_split)
    else:
        saved = json.loads(options.report.read_text())
        dataset, split = saved["metadata"]["reference_dataset"], saved["metadata"]["reference_split"]
    output_dir = options.output_dir or Path("outputs/joint_distribution_comparison") / (
        f"{dataset}_{split or 'full'}_{datetime.now():%Y-%m-%d_%H-%M-%S}")
    if options.report:
        # Keep existing analysis folders intact even if an overlapping output is requested.
        source = options.report.resolve()
        resolved_output = output_dir.resolve()
        if source.parent == resolved_output or source.is_relative_to(resolved_output):
            parser.error("--output-dir deve essere una nuova cartella separata dal report originale")
    else:
        source = output_dir / "analysis/report.json"
        with command_arguments([*analysis_options, "--output", str(source)]):
            analysis.main()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    report = json.loads(source.read_text())
    result = make_plots(report, output_dir, options.top_k, not options.no_pdf)
    result["source_report"] = str(source)
    result["source_report_sha256"] = source_hash
    (output_dir / "joint_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"Confronto Dataset + Base + MMFF + Atom details: {output_dir}")
    print(f"Panoramica: {output_dir / 'overview.png'}")


if __name__ == "__main__":
    main()
