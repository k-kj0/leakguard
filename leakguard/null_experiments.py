
"""Repeated null experiments for evaluating chance-level prediction.

The experiment generates balanced random labels and independent random
prediction scores. Since the scores contain no information about the
labels, the resulting AUC distribution provides a controlled null
reference for binary classification.

This is a synthetic statistical diagnostic, not a market prediction
benchmark or proof of trading profitability.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def _validate_parameters(n_observations, repetitions):
    if n_observations < 100:
        raise ValueError("Use at least 100 observations.")

    if repetitions < 20:
        raise ValueError("Use at least 20 repetitions.")


def _one_null_trial(rng, n_observations):
    """Generate independent random labels and prediction scores."""
    labels = np.zeros(n_observations, dtype=int)
    labels[: n_observations // 2] = 1
    rng.shuffle(labels)

    scores = rng.standard_normal(n_observations)

    return float(roc_auc_score(labels, scores))


def render_report(report):
    """Render the null experiment as a Markdown report."""
    summary = report["summary"]

    lines = [
        "# Leakguard Null Experiment Report",
        "",
        "## Configuration",
        "",
        f"- Observations per trial: **{report['n_observations']}**",
        f"- Repetitions: **{report['repetition_count']}**",
        f"- Random seed: **{report['seed']}**",
        "",
        "## Null AUC distribution",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Theoretical reference AUC | {summary['null_reference_auc']:.4f} |",
        f"| Mean observed AUC | {summary['mean_auc']:.4f} |",
        f"| Median observed AUC | {summary['median_auc']:.4f} |",
        f"| Standard deviation | {summary['std_auc']:.4f} |",
        f"| 2.5th percentile | {summary['p025_auc']:.4f} |",
        f"| 97.5th percentile | {summary['p975_auc']:.4f} |",
        f"| Fraction of trials with AUC >= 0.60 | {summary['fraction_auc_ge_060']:.1%} |",
        "",
        "## Interpretation",
        "",
        "The labels and prediction scores are generated independently. "
        "There is no predictive relationship in this experiment.",
        "",
        "The distribution describes the range of AUC values produced "
        "by this particular random-data setup. Individual trials can "
        "produce AUC values above or below 0.50 by chance.",
        "",
        "The 2.5th and 97.5th percentiles describe the simulated null "
        "distribution. They are not confidence intervals for a real "
        "market model.",
        "",
        "## Limitations",
        "",
        "- This is a synthetic independent-observation null.",
        "- It does not reproduce temporal dependence in market data.",
        "- It does not account for overlapping financial labels.",
        "- It does not model multiple testing; see the separate selection-bias experiment.",
        "- Results depend on sample size, repetition count and random seed.",
        "- A null simulation does not establish whether a real model has predictive value.",
        "",
    ]

    return "\n".join(lines)


def run_null_experiments(
    n_observations=1000,
    repetitions=500,
    seed=7,
    out_dir="results/null_experiments",
):
    """Run repeated independent-label null experiments."""
    _validate_parameters(n_observations, repetitions)

    rng = np.random.default_rng(seed)

    auc_values = [
        _one_null_trial(rng, n_observations)
        for _ in range(repetitions)
    ]

    auc_array = np.asarray(auc_values, dtype=float)

    summary = {
        "null_reference_auc": 0.5,
        "mean_auc": float(np.mean(auc_array)),
        "median_auc": float(np.median(auc_array)),
        "std_auc": float(np.std(auc_array, ddof=1)),
        "p025_auc": float(np.quantile(auc_array, 0.025)),
        "p975_auc": float(np.quantile(auc_array, 0.975)),
        "fraction_auc_ge_060": float(
            np.mean(auc_array >= 0.60)
        ),
        "trials_below_050": int(np.sum(auc_array < 0.5)),
        "trials_equal_050": int(np.sum(auc_array == 0.5)),
        "trials_above_050": int(np.sum(auc_array > 0.5)),
    }

    report = {
        "experiment": "independent_random_label_null",
        "version": 1,
        "seed": int(seed),
        "n_observations": int(n_observations),
        "repetition_count": int(repetitions),
        "summary": summary,
        "auc_values": [float(value) for value in auc_values],
        "interpretation": (
            "Synthetic independent-label null experiment. Prediction "
            "scores are independent of the binary labels. The simulated "
            "AUC distribution is a chance-level reference, not a "
            "confidence interval for a real model."
        ),
    }

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "latest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    markdown = render_report(report)

    (output / "latest.md").write_text(
        markdown,
        encoding="utf-8",
    )

    print(markdown)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--observations",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--out",
        default="results/null_experiments",
    )

    args = parser.parse_args()

    run_null_experiments(
        n_observations=args.observations,
        repetitions=args.repetitions,
        seed=args.seed,
        out_dir=args.out,
    )
