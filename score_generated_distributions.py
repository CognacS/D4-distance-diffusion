#!/usr/bin/env python3
"""Rank generated representations by distribution distances to a shared dataset.

Reads the existing multi-seed report; no molecular preprocessing or generation.
    python score_generated_distributions.py --report PATH/TO/report.json

The primary score is a declared, task-specific average of TVD distances over
five equally weighted descriptor families. This is a descriptive comparison
of measured marginals, not a test of equality of molecular distributions.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from compare_three_representations import COLORS, LABELS, REPRESENTATIONS, distributions


GROUPS = {
    "composition": ("atom_symbols", "formal_charges", "molecular_charge"),
    "atomic_environments": ("node_mmff_type", "node_hybridization", "node_is_in_ring", "node_is_aromatic"),
    "bonds": ("bond_types",),
    "molecular_structure": ("valid_num_atoms", "valid_num_bonds", "valid_num_rings", "valid_num_components"),
    "functional_groups": ("substructure_presence",),
}
GROUP_LABELS = {"composition": "Composizione e cariche", "atomic_environments": "Ambienti atomici",
                "bonds": "Legami", "molecular_structure": "Struttura molecolare",
                "functional_groups": "Gruppi funzionali"}
SOURCES = {
    "tvd": "https://www.stat.berkeley.edu/~aldous/RWG/Book_Ralph/Ch2.S4.html",
    "jensen_shannon": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html",
    "wasserstein": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html",
}
NUMERIC_FIELDS = {"formal_charges", "molecular_charge", "degrees", "total_hydrogens", "ring_sizes",
                  "graph_num_atoms", "graph_num_bonds", "graph_connected_components",
                  "valid_num_atoms", "valid_num_bonds", "valid_num_rings", "valid_num_components",
                  "valid_aromatic_ring_count"}


def summary(values):
    values = [float(value) for value in values if value is not None]
    return {"mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None, "n": len(values)}


def probability_arrays(reference, generated):
    """Align the union of categories, including categories absent on either side."""
    if not reference or not generated:
        return None
    keys = sorted(set(reference) | set(generated))
    p = np.array([reference.get(key, 0) for key in keys], dtype=float)
    q = np.array([generated.get(key, 0) for key in keys], dtype=float)
    if not np.isfinite(p).all() or not np.isfinite(q).all() or (p < 0).any() or (q < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative")
    if not p.sum() or not q.sum():
        return None
    return keys, p / p.sum(), q / q.sum()


def distribution_metrics(reference, generated, numeric=False):
    aligned = probability_arrays(reference, generated)
    if aligned is None:
        return {"tvd": None, "js_distance": None, "wasserstein": None}
    keys, p, q = aligned
    result = {"tvd": float(0.5 * np.abs(p - q).sum()),
              "js_distance": float(jensenshannon(p, q, base=2)), "wasserstein": None}
    if numeric:
        support = np.array([float(key) for key in keys])
        result["wasserstein"] = float(wasserstein_distance(support, support, u_weights=p, v_weights=q))
    return result


def presence_metrics(reference, generated):
    """Each functional-group presence is Bernoulli, not one category of a PMF."""
    if not reference or not generated:
        return {"tvd": None, "js_distance": None, "wasserstein": None}
    if set(reference) != set(generated):
        raise ValueError("Functional-group definitions differ between reference and generation")
    tvd, js = [], []
    for key in sorted(reference):
        p, q = reference[key], generated[key]
        if p is None or q is None:
            return {"tvd": None, "js_distance": None, "wasserstein": None}
        if not 0 <= p <= 1 or not 0 <= q <= 1:
            raise ValueError("Presence probabilities must be within [0,1]")
        tvd.append(abs(p - q))
        js.append(float(jensenshannon([p, 1 - p], [q, 1 - q], base=2)))
    return {"tvd": statistics.mean(tvd), "js_distance": statistics.mean(js), "wasserstein": None}


def composite_score(metrics, weights, metric):
    group_scores = {}
    for group, fields in GROUPS.items():
        values = [metrics.get(field, {}).get(metric) for field in fields]
        group_scores[group] = statistics.mean(values) if all(value is not None for value in values) else None
    active = [group for group, weight in weights.items() if weight > 0]
    missing = [group for group in active if group_scores[group] is None]
    score = None if missing else sum(weights[group] * group_scores[group] for group in active) / sum(weights.values())
    return score, group_scores, missing


def analyze_report(report, weights):
    if report.get("schema_version") != 2:
        raise ValueError("Expected schema_version=2 from compare_generated_distributions.py")
    if set(weights) != set(GROUPS) or any(not math.isfinite(v) or v < 0 for v in weights.values()) or not sum(weights.values()):
        raise ValueError("Weights must name the five groups, be finite/nonnegative, and have a positive sum")
    reference = distributions(report["reference"])
    expected = report["metadata"].get("expected_seeds", [0, 1, 2])
    result = {"metadata": report["metadata"], "sources": SOURCES,
              "score_definition": {"groups": GROUPS, "group_weights": weights,
                                   "normalized_group_weights": {key: value / sum(weights.values()) for key, value in weights.items()},
                                   "within_group_weights": "equal", "lower_is_better": True,
                                   "main_metric": "macro_tvd", "secondary_metric": "macro_js_distance"},
              "reference_quality": {key: report["reference"].get(key) for key in
                                    ["num_input_graphs", "num_valid_molecules", "mmff_coverage"]},
              "representations": {}, "warnings": list(report.get("warnings", []))}
    for name in REPRESENTATIONS:
        entry = report.get("representations", {}).get(name, {})
        runs = []
        seeds = [run["seed"] for run in entry.get("runs", [])]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Repeated seeds in {name}: {seeds}")
        missing_seeds = sorted(set(expected) - set(seeds))
        if missing_seeds:
            result["warnings"].append(f"{name}: seed mancanti {missing_seeds}")
        for run in entry.get("runs", []):
            generated = distributions(run["generated"])
            fields = sorted(set(reference) | set(generated))
            metrics = {field: (presence_metrics(reference.get(field, {}), generated.get(field, {}))
                               if field == "substructure_presence" else
                               distribution_metrics(reference.get(field, {}), generated.get(field, {}), field in NUMERIC_FIELDS))
                       for field in fields}
            tvd, group_tvd, missing = composite_score(metrics, weights, "tvd")
            js, group_js, _ = composite_score(metrics, weights, "js_distance")
            if missing:
                result["warnings"].append(f"{name}/seed={run['seed']}: indice complessivo indefinito, gruppi mancanti {missing}")
            quality = {key: run["generated"].get(key) for key in
                       ["num_input_graphs", "num_valid_molecules", "valid_fraction", "uniqueness_fraction", "mmff_coverage"]}
            if quality["mmff_coverage"] and quality["mmff_coverage"]["failed_valid_molecules"]:
                result["warnings"].append(f"{name}/seed={run['seed']}: MMFF disponibile su un sottoinsieme delle molecole valide")
            runs.append({"seed": run["seed"], "generated_file": run.get("generated_file"), "quality": quality,
                         "descriptor_metrics": metrics, "macro_tvd": tvd, "macro_js_distance": js,
                         "group_tvd": group_tvd, "group_js_distance": group_js})
        descriptor_names = sorted({field for run in runs for field in run["descriptor_metrics"]})
        stats = {
            "macro_tvd": summary([run["macro_tvd"] for run in runs]),
            "macro_js_distance": summary([run["macro_js_distance"] for run in runs]),
            "group_tvd": {group: summary([run["group_tvd"][group] for run in runs]) for group in GROUPS},
            "group_js_distance": {group: summary([run["group_js_distance"][group] for run in runs]) for group in GROUPS},
            "valid_fraction": summary([run["quality"]["valid_fraction"] for run in runs]),
            "descriptors": {field: {metric: summary([run["descriptor_metrics"].get(field, {}).get(metric) for run in runs])
                                    for metric in ["tvd", "js_distance", "wasserstein"]} for field in descriptor_names},
        }
        result["representations"][name] = {"available_seeds": seeds, "missing_seeds": missing_seeds, "runs": runs, "summary": stats}
    for metric in ["macro_tvd", "macro_js_distance"]:
        result["ranking_" + metric] = sorted(
            [name for name in REPRESENTATIONS if result["representations"][name]["summary"][metric]["mean"] is not None],
            key=lambda name: result["representations"][name]["summary"][metric]["mean"])
    if not result["ranking_macro_tvd"]:
        raise ValueError("No model has a complete, comparable composite score; inspect missing distributions")
    # A descriptive sensitivity check, not a significance test or a seed-pairing assumption.
    sensitivity = {}
    active_groups = [group for group, weight in weights.items() if weight > 0]
    if len(active_groups) > 1:
        for omitted in active_groups:
            remaining = [group for group in active_groups if group != omitted]
            means = {}
            for name, entry in result["representations"].items():
                values = [sum(weights[g] * run["group_tvd"][g] for g in remaining) / sum(weights[g] for g in remaining)
                          for run in entry["runs"] if run["macro_tvd"] is not None
                          and all(run["group_tvd"][g] is not None for g in remaining)]
                if values:
                    means[name] = statistics.mean(values)
            order = sorted(means, key=means.get)
            sensitivity[omitted] = {"ranking": order, "means": means}
    result["leave_one_group_out"] = sensitivity
    return result


def format_stat(stat):
    if stat["mean"] is None:
        return "n/d"
    return f"{stat['mean']:.6f}" + (f" ± {stat['std']:.6f}" if stat["std"] is not None else " (1 seed)")


def save_figure(figure, directory, name):
    for extension in ["png", "pdf"]:
        figure.savefig(directory / f"{name}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(figure)


def available_wasserstein_fields(result):
    return sorted(field for field in NUMERIC_FIELDS if any(
        entry["summary"]["descriptors"].get(field, {}).get("wasserstein", {}).get("n", 0)
        for entry in result["representations"].values()))


def wasserstein_markdown_lines(result):
    """Show every available numeric distance in the human-readable score report."""
    lines = ["", "## Wasserstein-1 per descrittore", "",
             "Valori più bassi indicano maggiore somiglianza per quel descrittore. "
             "Le celle riportano media ± deviazione standard campionaria e numero di seed di training.", "",
             "Le unità dipendono dal descrittore: non si sommano queste distanze in un unico punteggio.", ""]
    fields = available_wasserstein_fields(result)
    if not fields:
        return lines + ["Nessuna distanza Wasserstein calcolabile dai dati disponibili.", ""]
    lines += ["| Descrittore | Unità | Base | MMFF | Atom details | Minimo medio |",
              "|---|---|---|---|---|---|"]
    for field in fields:
        stats = {name: result["representations"].get(name, {}).get("summary", {}).get("descriptors", {})
                 .get(field, {}).get("wasserstein", {}) for name in REPRESENTATIONS}
        means = {name: stat["mean"] for name, stat in stats.items() if stat.get("mean") is not None}
        minimum = min(means.values()) if means else None
        best = [LABELS[name] for name, value in means.items()
                if math.isclose(value, minimum, abs_tol=1e-12, rel_tol=0)]
        cells = []
        for name in REPRESENTATIONS:
            stat = stats[name]
            if stat.get("mean") is None:
                cells.append("n/d (n=0)")
            elif stat.get("std") is None:
                cells.append(f"{stat['mean']:.6f}; SD n/d (n={stat['n']})")
            else:
                cells.append(f"{stat['mean']:.6f} ± {stat['std']:.6f} (n={stat['n']})")
        unit = "e" if field in {"formal_charges", "molecular_charge"} else "conteggio"
        lines.append("| " + " | ".join([field, unit, *cells, ", ".join(best) or "n/d"]) + " |")
    lines += ["", "`e` indica l'unità di carica elementare. Il minimo medio è un confronto descrittivo, "
              "non una dichiarazione di significatività statistica. Con un solo seed la SD è indefinita.", "",
              "Dettagli: [riepilogo CSV](wasserstein_summary.csv), [valori per seed](wasserstein_by_seed.csv), "
              "[grafico PNG](wasserstein.png), [grafico PDF](wasserstein.pdf).", ""]
    return lines


def write_wasserstein_outputs(result, directory):
    """Export numeric distances separately; never interpret MMFF IDs as numbers."""
    directory.mkdir(parents=True, exist_ok=True)
    fields = available_wasserstein_fields(result)
    with (directory / "wasserstein_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["representation", "descriptor", "wasserstein_mean", "wasserstein_std", "n_seeds", "unit"])
        for field in fields:
            for name, entry in result["representations"].items():
                stat = entry["summary"]["descriptors"].get(field, {}).get("wasserstein", {})
                writer.writerow([name, field, stat.get("mean"), stat.get("std"), stat.get("n", 0),
                                 "elementary_charge" if field in {"formal_charges", "molecular_charge"} else "count"])
    with (directory / "wasserstein_by_seed.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["representation", "seed", "descriptor", "wasserstein"])
        for name, entry in result["representations"].items():
            for run in entry["runs"]:
                for field in fields:
                    writer.writerow([name, run["seed"], field, run["descriptor_metrics"].get(field, {}).get("wasserstein")])
    if not fields:
        return
    rows = math.ceil(len(fields) / 2)
    figure, axes = plt.subplots(rows, 2, figsize=(13, rows * 2.6), squeeze=False, layout="constrained")
    for ax, field in zip(axes.flat, fields):
        names = [name for name, entry in result["representations"].items()
                 if entry["summary"]["descriptors"].get(field, {}).get("wasserstein", {}).get("n", 0)]
        upper = 0.
        for index, name in enumerate(names):
            entry = result["representations"][name]
            stat = entry["summary"]["descriptors"][field]["wasserstein"]
            values = [run["descriptor_metrics"].get(field, {}).get("wasserstein") for run in entry["runs"]]
            values = [value for value in values if value is not None]
            ax.scatter(values, [index + offset for offset in np.linspace(-.1, .1, len(values))],
                       color=COLORS[name], s=18, alpha=.6)
            ax.errorbar(stat["mean"], index, xerr=stat["std"], fmt="D", color=COLORS[name], capsize=3)
            upper = max(upper, max(values), stat["mean"] + (stat["std"] or 0))
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([LABELS[name] for name in names], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(upper * 1.15, .0001))
        ax.set_title(field.replace("_", " "), fontsize=10)
        ax.set_xlabel("Wasserstein (unità di carica)" if field in {"formal_charges", "molecular_charge"}
                      else "Wasserstein (unità del conteggio)", fontsize=8)
        ax.grid(axis="x", alpha=.2)
    for ax in list(axes.flat)[len(fields):]:
        ax.set_visible(False)
    figure.suptitle("Wasserstein-1 dal dataset: più bassa = più simile\n"
                    "Media ± deviazione standard tra seed; scale indipendenti, nessuna somma tra descrittori", fontsize=12)
    save_figure(figure, directory, "wasserstein")


def draw_plots(result, directory):
    order = result["ranking_macro_tvd"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout="constrained")
    for ax, metric, title in zip(axes, ["macro_tvd", "macro_js_distance"], ["TVD complessiva (principale)", "Jensen–Shannon (secondaria)"]):
        upper = 0
        for index, name in enumerate(order):
            entry = result["representations"][name]
            stat = entry["summary"][metric]
            values = [run[metric] for run in entry["runs"] if run[metric] is not None]
            if not values:
                continue
            ax.scatter(values, [index + offset for offset in np.linspace(-.12, .12, len(values))],
                       color=COLORS[name], s=30, alpha=.65)
            ax.errorbar(stat["mean"], index, xerr=stat["std"] if stat["std"] is not None else None,
                        fmt="D", color=COLORS[name], capsize=5, markersize=7, linewidth=2)
            upper = max(upper, max(values), stat["mean"] + (stat["std"] or 0))
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"{LABELS[name]} (n={result['representations'][name]['summary'][metric]['n']})" for name in order])
        ax.invert_yaxis()
        ax.set_xlim(0, max(upper * 1.2, .001))
        ax.set_xlabel("Distanza dal dataset: più bassa = più simile")
        ax.set_title(title)
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"{result['metadata']['reference_dataset']} · {result['metadata']['reference_split'] or 'full'}\n"
                    "Punti = singoli seed; rombo e barra = media ± deviazione standard", fontsize=12)
    save_figure(figure, directory, "ranking")

    for filename, fields, group_mode in [
        ("groups_tvd", list(GROUPS), True),
        ("descriptors_tvd", sorted({field for entry in result["representations"].values()
                                   for field in entry["summary"]["descriptors"]}), False),
    ]:
        matrix = np.array([[result["representations"][name]["summary"]["group_tvd"][field]["mean"]
                            if group_mode else result["representations"][name]["summary"]["descriptors"].get(field, {}).get("tvd", {}).get("mean")
                            for name in order] for field in fields], dtype=float)
        figure, ax = plt.subplots(figsize=(9, max(4, len(fields) * .45 + 1.5)), layout="constrained")
        maximum = max(float(np.nanmax(matrix)), .001)
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap="YlOrRd", vmin=0,
                          vmax=maximum, aspect="auto")
        for row in range(len(fields)):
            finite = matrix[row][np.isfinite(matrix[row])]
            best = finite.min() if len(finite) else None
            for column in range(len(order)):
                value = matrix[row, column]
                if np.isfinite(value):
                    ax.text(column, row, f"{value:.5f}", ha="center", va="center", fontsize=9,
                            color="white" if value > .6 * maximum else "black")
                    if math.isclose(value, best, abs_tol=1e-12, rel_tol=0):
                        ax.add_patch(plt.Rectangle((column - .48, row - .45), .96, .9, fill=False,
                                                   edgecolor="#226644", linewidth=2))
                else:
                    ax.text(column, row, "n/d", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([LABELS[name] for name in order])
        ax.set_yticks(range(len(fields)))
        ax.set_yticklabels([GROUP_LABELS[field] if group_mode else field for field in fields])
        ax.set_title("TVD media dal dataset · più bassa = più simile\nBordo verde = minimo della riga (descrittivo)")
        figure.colorbar(image, ax=ax, label="TVD", shrink=.7)
        save_figure(figure, directory, filename)


def write_outputs(result, directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "ranking.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["rank", "representation", "macro_tvd_mean", "macro_tvd_std", "n_seeds", "macro_js_mean", "macro_js_std", "valid_fraction_mean"])
        for rank, name in enumerate(result["ranking_macro_tvd"], 1):
            stat = result["representations"][name]["summary"]
            writer.writerow([rank, name, stat["macro_tvd"]["mean"], stat["macro_tvd"]["std"], stat["macro_tvd"]["n"],
                             stat["macro_js_distance"]["mean"], stat["macro_js_distance"]["std"], stat["valid_fraction"]["mean"]])
    with (directory / "distances_by_seed.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["representation", "seed", "descriptor", "tvd", "js_distance", "wasserstein"])
        for name, entry in result["representations"].items():
            for run in entry["runs"]:
                for field, metric in run["descriptor_metrics"].items():
                    writer.writerow([name, run["seed"], field, metric["tvd"], metric["js_distance"], metric["wasserstein"]])
    with (directory / "group_scores.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["representation", "group", "tvd_mean", "tvd_std", "n_seeds", "normalized_weight"])
        for name, entry in result["representations"].items():
            for group, stat in entry["summary"]["group_tvd"].items():
                writer.writerow([name, group, stat["mean"], stat["std"], stat["n"], result["score_definition"]["normalized_group_weights"][group]])
    draw_plots(result, directory)
    write_wasserstein_outputs(result, directory)
    (directory / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ["# Distanze dal dataset di riferimento", "",
             f"Dataset: **{result['metadata']['reference_dataset']}**, split **{result['metadata']['reference_split'] or 'full'}**.", "",
             "Valori più bassi indicano distribuzioni misurate più simili al riferimento.", "",
             "| Posizione | Rappresentazione | TVD complessiva (media ± SD) | Seed | Jensen–Shannon | Validità media |",
             "|---|---|---|---|---|---|"]
    for rank, name in enumerate(result["ranking_macro_tvd"], 1):
        stat = result["representations"][name]["summary"]
        lines.append(f"| {rank} | {LABELS[name]} | {format_stat(stat['macro_tvd'])} | {stat['macro_tvd']['n']} | "
                     f"{format_stat(stat['macro_js_distance'])} | {100 * stat['valid_fraction']['mean']:.3f}% |")
    lines += ["", "La graduatoria riguarda questo indice descrittivo e questi pesi. Non dimostra una superiorità statisticamente significativa."]
    lines += wasserstein_markdown_lines(result)
    lines += ["## Come è costruito l'indice", "",
              "Per ogni seed si calcola prima la distanza per proprietà, poi la media nelle famiglie e infine la media pesata delle famiglie.",
              "Solo dopo si calcolano media e deviazione standard campionaria (n−1) tra seed. Ogni seed pesa uguale.", ""]
    for group, fields in GROUPS.items():
        weight = result["score_definition"]["normalized_group_weights"][group]
        lines.append(f"- **{GROUP_LABELS[group]}**, peso {weight:.1%}: {', '.join(fields)}.")
    lines += ["", "TVD = ½ Σ|p−q|, tra 0 e 1. L'aggregazione tra famiglie è una scelta di questa analisi, non uno standard universale.",
              "Per i gruppi funzionali si media |p−q| sulle singole presenze Bernoulli: non si normalizzano insieme gruppi sovrapponibili.",
              "Jensen–Shannon è la radice della divergenza, con logaritmi base 2, anch'essa tra 0 e 1.",
              "Wasserstein è riportata per le quantità numeriche nelle loro unità e resta fuori dall'indice complessivo.", "",
              "CSV dedicati: `wasserstein_summary.csv` (media/SD) e `wasserstein_by_seed.csv`; grafici: `wasserstein.png/.pdf`.", "",
              "## Sensibilità ai pesi", ""]
    for group, check in result["leave_one_group_out"].items():
        lines.append(f"- Escludendo {GROUP_LABELS[group]}: {' → '.join(LABELS[name] for name in check['ranking'])}.")
    lines += ["", "## Limiti e copertura", "",
              "Sono confrontate distribuzioni marginali di descrittori, non la distribuzione completa delle strutture molecolari.",
              "Le distanze empiriche dipendono anche dalla numerosità dei campioni; confrontare conteggi e seed tra modelli.",
              "Le proprietà chimiche si riferiscono ai frammenti maggiori validi; validità e copertura MMFF sono riportate separatamente.",
              "SD tra pochi seed non è un intervallo di confidenza. Non si assume che seed con lo stesso numero siano osservazioni appaiate.",
              "Descrittori non calcolabili non vengono trattati come distanza zero. Un seed con una famiglia attiva incompleta non riceve un indice complessivo.",
              "Un indice di 0.01 non significa che il 99% delle molecole siano corrette o identiche al dataset.", "", "## Fonti", ""]
    lines += [f"- [{key}]({url})" for key, url in SOURCES.items()]
    if result["warnings"]:
        lines += ["", "## Avvisi", ""] + [f"- {message}" for message in result["warnings"]]
    (directory / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=Path("outputs/distribution_comparison/qm9_three_representations/report.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--group-weights", nargs="+", default=[], metavar="GROUP=WEIGHT",
                        help="Override composition, atomic_environments, bonds, molecular_structure, functional_groups (default 1 each)")
    args = parser.parse_args()
    weights = dict.fromkeys(GROUPS, 1.)
    for option in args.group_weights:
        group, _, value = option.partition("=")
        try:
            if group not in weights:
                raise ValueError()
            weights[group] = float(value)
        except ValueError:
            parser.error(f"Invalid group weight: {option}")
    source_bytes = args.report.read_bytes()
    report = json.loads(source_bytes)
    result = analyze_report(report, weights)
    output = args.output_dir or Path("outputs/distribution_scores") / (
        f"{report['metadata']['reference_dataset']}_{report['metadata']['reference_split'] or 'full'}_{datetime.now():%Y-%m-%d_%H-%M-%S}")
    source = args.report.resolve()
    if source.parent == output.resolve() or source.is_relative_to(output.resolve()):
        parser.error("Choose a separate output folder to preserve the source analysis")
    result["source_report"] = str(source)
    result["source_report_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    write_outputs(result, output)
    print(f"Riferimento: {report['metadata']['reference_dataset']}/{report['metadata']['reference_split'] or 'full'}")
    print("Graduatoria TVD complessiva, più bassa = più simile (media ± SD tra seed):")
    for rank, name in enumerate(result["ranking_macro_tvd"], 1):
        stat = result["representations"][name]["summary"]["macro_tvd"]
        print(f"  {rank}. {LABELS[name]}: {format_stat(stat)} (n={stat['n']})")
    for message in result["warnings"]:
        print("WARNING:", message, file=sys.stderr)
    print(f"Report leggibile: {output / 'summary.md'}")


if __name__ == "__main__":
    main()
