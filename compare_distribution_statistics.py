#!/usr/bin/env python3
"""Permutation tests of model-score differences, using training seeds as replicates.

Examples:
    python compare_distribution_statistics.py --scores PATH/TO/scores/metrics.json
    python compare_distribution_statistics.py --scores train=TRAIN.json test=TEST.json

References are tested separately. Equal seed numbers across models are not
treated as paired observations. Molecules and atoms are not training replicates.
"""

import argparse
import csv
from datetime import datetime
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from score_generated_distributions import GROUPS, LABELS, NUMERIC_FIELDS


def holm_adjust(pvalues):
    """Step-down family-wise correction, preserving input order."""
    order = sorted(range(len(pvalues)), key=lambda index: pvalues[index])
    adjusted = [None] * len(pvalues)
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1., running)
    return adjusted


def partitions(indices, sizes):
    if len(sizes) == 1:
        yield [tuple(indices)]
        return
    for selected in combinations(indices, sizes[0]):
        chosen = set(selected)
        remaining = tuple(index for index in indices if index not in chosen)
        for rest in partitions(remaining, sizes[1:]):
            yield [selected, *rest]


def mean_difference(*samples):
    return abs(float(np.mean(samples[0]) - np.mean(samples[1])))


def anova_f(*samples):
    if max(float(sample.max()) for sample in samples) == min(float(sample.min()) for sample in samples):
        return 0.
    grand = sum(float(sample.sum()) for sample in samples) / sum(map(len, samples))
    between = sum(len(sample) * (float(sample.mean()) - grand) ** 2 for sample in samples)
    within = sum(float(np.square(sample - sample.mean()).sum()) for sample in samples)
    if within == 0:
        return math.inf if between > 0 else 0.
    return between / (len(samples) - 1) / (within / (sum(map(len, samples)) - len(samples)))


def permutation_comparison(samples, omnibus=False, max_permutations=20000, rng_seed=0):
    samples = [np.asarray(sample, dtype=float) for sample in samples]
    if len(samples) < 2 or any(len(sample) < 2 for sample in samples):
        raise ValueError("At least two independent seeds per model are required")
    if any(not np.isfinite(sample).all() for sample in samples):
        raise ValueError("Nonfinite seed scores")
    sizes = tuple(map(len, samples))
    total = math.prod(math.comb(sum(sizes[index:]), sizes[index]) for index in range(len(sizes) - 1))
    statistic = anova_f if omnibus else mean_difference
    observed = statistic(*samples)
    pooled = np.concatenate(samples)
    exact = total <= max_permutations
    if exact:
        allocations = partitions(tuple(range(len(pooled))), sizes)
    else:
        generator = np.random.default_rng(rng_seed)
        cutoffs = np.cumsum(sizes)[:-1]
        allocations = (np.split(generator.permutation(len(pooled)), cutoffs) for _ in range(max_permutations))
    extreme = 0
    count = 0
    tolerance = 100 * np.finfo(float).eps * max(abs(observed), 1e-15) if math.isfinite(observed) else 0
    for allocation in allocations:
        value = statistic(*(pooled[list(group)] for group in allocation))
        extreme += (math.isinf(value) if math.isinf(observed) else value >= observed - tolerance)
        count += 1
    pvalue = extreme / count if exact else (extreme + 1) / (count + 1)
    floor = (2 / total if not omnibus and len(set(sizes)) == 1 else 1 / total) if exact else 1 / (count + 1)
    return {"statistic": observed if math.isfinite(observed) else "infinity", "p_raw": pvalue,
            "method": "exact_permutation" if exact else "monte_carlo_permutation",
            "n_permutations": count, "n_distinct_partitions": total,
            "p_resolution_lower_bound": floor, "sample_sizes": sizes}


def target_values(run, target):
    if target in {"macro_tvd", "macro_js_distance"}:
        return run.get(target)
    kind, field = target.split(":", 1)
    if kind == "group_tvd":
        return run["group_tvd"].get(field)
    return run["descriptor_metrics"].get(field, {}).get("wasserstein")


def analyze_inputs(inputs, alpha=.05, max_permutations=20000, rng_seed=0):
    rows, warnings, contexts = [], [], {}
    targets = ["macro_tvd", "macro_js_distance"] + ["group_tvd:" + key for key in GROUPS]
    targets += ["wasserstein:" + key for key in sorted(NUMERIC_FIELDS)]
    for label, report in inputs.items():
        contexts[label] = {"metadata": report["metadata"], "score_definition": report["score_definition"]}
        for target in targets:
            samples = {}
            seeds = {}
            for name, entry in report["representations"].items():
                values = [(run["seed"], target_values(run, target)) for run in entry["runs"]]
                values = [(seed, value) for seed, value in values if value is not None]
                if len({seed for seed, _ in values}) != len(values):
                    raise ValueError(f"Duplicate seeds in {label}/{name}")
                if len(values) >= 2:
                    seeds[name] = [seed for seed, _ in values]
                    samples[name] = [value for _, value in values]
            if len(samples) < 2:
                warnings.append(f"{label}/{target}: meno di due modelli con almeno due seed; test non eseguito")
                continue
            family = "primary" if target == "macro_tvd" else "exploratory"
            if len(samples) >= 3:
                values = permutation_comparison(list(samples.values()), True, max_permutations, rng_seed)
                rows.append({"reference": label, "target": target, "family": family, "comparison": "omnibus",
                             "model_a": "+".join(samples), "model_b": "", "mean_a": None, "mean_b": None,
                             "mean_difference_a_minus_b": None, "cliffs_delta_lower_is_better": None,
                             "seed_ids": seeds, **values})
            for name_a, name_b in combinations(samples, 2):
                a, b = samples[name_a], samples[name_b]
                values = permutation_comparison([a, b], False, max_permutations, rng_seed)
                cliff = sum((x < y) - (x > y) for x in a for y in b) / (len(a) * len(b))
                rows.append({"reference": label, "target": target, "family": family, "comparison": "pairwise",
                             "model_a": name_a, "model_b": name_b, "mean_a": statistics.mean(a), "mean_b": statistics.mean(b),
                             "mean_difference_a_minus_b": statistics.mean(a) - statistics.mean(b),
                             "cliffs_delta_lower_is_better": cliff,
                             "seed_ids": {name_a: seeds[name_a], name_b: seeds[name_b]}, **values})
    for family in ["primary", "exploratory"]:
        selected = [row for row in rows if row["family"] == family]
        for row, adjusted in zip(selected, holm_adjust([row["p_raw"] for row in selected])):
            row["p_holm"] = adjusted
            row["reject_at_alpha"] = adjusted <= alpha
            row["holm_family_size"] = len(selected)
    return {"alpha": alpha, "contexts": contexts, "tests": rows, "warnings": warnings,
            "notes": [
                "Independent training seeds are the experimental units, conditional on the fixed reference dataset.",
                "Equal numeric seed IDs across models do not establish paired observations.",
                "Null hypothesis: exchangeable seed-score distributions across models; the pairwise statistic is the absolute mean difference.",
                "Omnibus statistic is ANOVA F with a permutation null, not a parametric F-distribution approximation.",
                "Holm families span all input references: primary includes macro-TVD omnibus and pairwise tests; all other tests form a separate exploratory family.",
                "Pairwise exact two-sided tests with three seeds per model have minimum p=2/binomial(6,3)=0.10 before adjustment.",
                "Failing to reject does not establish equivalence. More independent training seeds are needed for greater power.",
                "References are evaluated separately; this script does not test train versus test or re-estimate molecule-level sampling uncertainty.",
                "Different reference sample sizes change empirical-distance noise. Train and test scores cannot be interpreted as identically calibrated samples.",
                "Cliff's delta is positive when model A has lower distances more often than model B. It is descriptive, not a p-value.",
            ], "sources": {
                "permutation_tests": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html",
                "holm": "https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html"}}


def write_outputs(result, directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "statistics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    if result["tests"]:
        columns = [key for key in result["tests"][0] if key not in {"seed_ids", "sample_sizes"}]
        with (directory / "statistical_tests.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows({key: row[key] for key in columns} for row in result["tests"])
    lines = ["# Test statistici sulle differenze tra modelli", "",
             "Unità sperimentale: seed indipendente di training, con riferimento fissato. Soglia alpha: " + str(result["alpha"]) + ".", "",
             "Test di permutazione; confronti a coppie bilaterali. Holm corregge insieme i confronti primari di tutti i riferimenti forniti.", "",
             "| Riferimento | Confronto TVD complessiva | Differenza media A−B | p grezzo | p Holm | Rifiuto H0 |",
             "|---|---|---:|---:|---:|---|"]
    primary = [row for row in result["tests"] if row["family"] == "primary"]
    for row in primary:
        label = "Omnibus (tre modelli)" if row["comparison"] == "omnibus" else LABELS[row["model_a"]] + " vs " + LABELS[row["model_b"]]
        difference = "—" if row["mean_difference_a_minus_b"] is None else f"{row['mean_difference_a_minus_b']:.7f}"
        lines.append(f"| {row['reference']} | {label} | {difference} | {row['p_raw']:.4f} | {row['p_holm']:.4f} | "
                     + ("Sì" if row["reject_at_alpha"] else "No") + " |")
    lines += ["", "Una differenza media negativa favorisce A, perché una distanza più bassa è migliore.", "",
              "**Con tre seed contro tre il p-value esatto bilaterale minimo è 0.10:** questi confronti a coppie non possono rifiutare H0 a 0.05.",
              "Un test omnibus può avere una risoluzione diversa; un eventuale risultato globale non identifica da solo la coppia differente.",
              "Nessun rifiuto non significa che i modelli siano equivalenti: la potenza con tre seed è limitata.", "",
              "## Dettagli", "",
              "Il CSV include test esplorativi sulle famiglie TVD, Jensen–Shannon complessiva e Wasserstein per descrittore numerico.",
              "Gli esplorativi hanno una famiglia Holm separata. I punteggi dei singoli descrittori non sono repliche di training.",
              "Le permutazioni sono esatte quando fattibili; altrimenti si usa Monte Carlo con correzione (conteggio+1)/(permutazioni+1).", "",
              "Questi test confrontano modelli sullo stesso riferimento. Train e test sono analizzati separatamente; "
              "non è un test diretto train-vs-test né un test sulle distribuzioni molecolari complete.",
              "Le diverse numerosità dei riferimenti influenzano le distanze empiriche e la loro variabilità.", ""]
    lines += [f"- [{name}]({url})" for name, url in result["sources"].items()]
    lines += ["", "## Avvisi", ""] + [f"- {message}" for message in result["warnings"]]
    (directory / "summary.md").write_text("\n".join(lines) + "\n")
    pairwise = [row for row in primary if row["comparison"] == "pairwise"]
    if pairwise:
        figure, ax = plt.subplots(figsize=(10, max(4, len(pairwise) * .55)), layout="constrained")
        x = np.arange(len(pairwise))
        ax.barh(x - .15, [row["p_raw"] for row in pairwise], .3, label="p grezzo", color="#4477AA")
        ax.barh(x + .15, [row["p_holm"] for row in pairwise], .3, label="p Holm", color="#888888")
        ax.axvline(result["alpha"], color="#AA3333", linestyle="--", label="alpha")
        ax.set_yticks(x)
        ax.set_yticklabels([f"{row['reference']}: {LABELS[row['model_a']]} / {LABELS[row['model_b']]}" for row in pairwise])
        ax.set_xlim(0, 1.05)
        ax.invert_yaxis()
        ax.set_xlabel("p-value: non è la probabilità che H0 sia vera")
        ax.set_title("Confronti bilaterali della TVD complessiva tra modelli")
        ax.legend(loc="lower right")
        for suffix in ["png", "pdf"]:
            figure.savefig(directory / f"primary_pvalues.{suffix}", dpi=170)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", nargs="+", required=True, help="metrics.json files, optionally LABEL=PATH")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--max-permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0, help="Monte Carlo RNG seed; no effect on exact tests")
    args = parser.parse_args()
    if not 0 < args.alpha < 1 or args.max_permutations < 1:
        parser.error("alpha must be between 0 and 1 and max-permutations positive")
    inputs, sources = {}, {}
    for argument in args.scores:
        label, separator, value = argument.partition("=")
        path = Path(value if separator else argument)
        raw = path.read_bytes()
        report = json.loads(raw)
        label = label if separator else (report["metadata"]["reference_split"] or "full")
        if label in inputs:
            parser.error(f"Duplicate label {label}; use unique LABEL=PATH arguments")
        inputs[label] = report
        sources[label] = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
    result = analyze_inputs(inputs, args.alpha, args.max_permutations, args.seed)
    result["input_files"] = sources
    result["monte_carlo_seed"] = args.seed
    output = args.output_dir or Path("outputs/distribution_statistics") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if any(Path(source["path"]).parent == output.resolve() for source in sources.values()):
        parser.error("Use a separate output directory to preserve the score summaries")
    write_outputs(result, output)
    print(f"Risultati: {output / 'summary.md'}")
    for row in result["tests"]:
        if row["family"] == "primary":
            print(row["reference"], row["comparison"], row["model_a"], row["model_b"],
                  f"p={row['p_raw']:.4f}, p_Holm={row['p_holm']:.4f}, reject={row['reject_at_alpha']}")


if __name__ == "__main__":
    main()
