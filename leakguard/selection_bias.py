
"""Null-model experiment demonstrating selection bias.

Every candidate model is evaluated on random scores unrelated to
random binary labels. Selecting the best result from many trials
can produce an apparently impressive AUC by chance alone.

This is a controlled statistical demonstration, not a trading
strategy or an estimate of real-world model performance.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def _validate_parameters(n_observations, n_trials, repetitions):
    if n_observations < 4:
        raise ValueError("n_observations must be at least 4.")

    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")

    if repetitions < 1:
        raise ValueError("repetitions must be at least 1.")


def _run_single_repetition(rng, n_observations, n_trials):
    """Evaluate independent random predictors and retain the best AUC."""
    labels = np.zeros(n_observations, dtype=int)

    labels[: n_observations // 2] = 1
    rng.shuffle(labels)

    trial_aucs = []

    for _ in range(n_trials):
        random_scores = rng.standard_normal(n_observations)

        auc = float(
            roc_auc_score(labels, random_scores)
        )

        trial_aucs.append(auc)

    selected_auc = max(trial_aucs)

    return {
        "trial_count": n_trials,
        "selected_auc": float(selected_auc),
        "mean_trial_auc": float(np.mean(trial_aucs)),
        "best_trial_auc": float(selected_auc),
        "worst_trial_auc": float(min(trial_aucs)),
    }


def render_report(report):
    """Render the experiment results as Markdown."""
    summary = report["summary"]

    lines = [
        "# Leakguard Selection-Bias Experiment",
        "",
        "## Experiment configuration",
        "",
        f"- Observations per repetition: {report['n_observations']}",
        f"- Independent random trials: {report['n_trials']}",
        f"- Repetitions: {report['repetition_count']}",
        f"- Random seed: {report['seed']}",
        "",
        "## Results",
        "",
        f"- Null reference AUC: **{summary['null_reference_auc']:.4f}**",
        f"- Mean selected AUC: **{summary['mean_selected_auc']:.4f}**",
        f"- Median selected AUC: **{summary['median_selected_auc']:.4f}**",
        f"- 95th percentile selected AUC: **{summary['p95_selected_auc']:.4f}**",
        f"- Mean unselected trial AUC: **{summary['mean_trial_auc']:.4f}**",
        f"- Mean selection uplift: **{summary['selection_uplift']:.4f}**",
        "",
        "## Interpretation",
        "",
        "Every candidate score in this experiment is independent of the "
        "binary outcome labels. There is no genuine predictive signal.",
        "",
        "Selecting the highest AUC from many random trials can nevertheless "
        "produce an apparently strong result. Repeating the experiment "
        "shows how this selection effect behaves under the specified "
        "simulation settings.",
        "",
        "This experiment does not estimate the performance of a real "
        "market model. It does not reproduce every form of financial "
        "multiple testing, dependence, or non-stationarity.",
        "",
    ]

    return "\n".join(lines)


def run_selection_bias(
    n_observations=300,
    n_trials=20,
    repetitions=30,
    seed=7,
    out_dir="results/selection_bias",
):
    """Run repeated null experiments and report the selection effect."""
    _validate_parameters(
        n_observations,
        n_trials,
        repetitions,
    )

    rng = np.random.default_rng(seed)

    results = []

    for repetition in range(repetitions):
        result = _run_single_repetition(
            rng,
            n_observations,
            n_trials,
        )

        result["repetition"] = repetition + 1
        results.append(result)

    selected_aucs = np.array(
        [item["selected_auc"] for item in results],
        dtype=float,
    )

    mean_trial_aucs = np.array(
        [item["mean_trial_auc"] for item in results],
        dtype=float,
    )

    null_reference_auc = 0.5
    mean_selected_auc = float(np.mean(selected_aucs))

    summary = {
        "null_reference_auc": null_reference_auc,
        "mean_selected_auc": mean_selected_auc,
        "median_selected_auc": float(np.median(selected_aucs)),
        "p95_selected_auc": float(
            np.percentile(selected_aucs, 95)
        ),
        "mean_trial_auc": float(np.mean(mean_trial_aucs)),
        "selection_uplift": float(
            mean_selected_auc - null_reference_auc
        ),
        "repetitions_above_null": int(
            np.sum(selected_aucs > null_reference_auc)
        ),
    }

    report = {
        "experiment": "random_predictor_selection_bias",
        "version": 1,
        "seed": int(seed),
        "n_observations": int(n_observations),
        "n_trials": int(n_trials),
        "repetition_count": int(repetitions),
        "repetitions": results,
        "summary": summary,
        "interpretation": (
            "Controlled null simulation. Random predictors have no "
            "relationship with the labels. Selecting the best result "
            "among multiple trials demonstrates selection bias."
        ),
    }

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "latest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    (output / "latest.md").write_text(
        render_report(report),
        encoding="utf-8",
    )

    print(render_report(report))

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--n-observations", type=int, default=300)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out",
        default="results/selection_bias",
    )

    args = parser.parse_args()

    run_selection_bias(
        n_observations=args.n_observations,
        n_trials=args.n_trials,
        repetitions=args.repetitions,
        seed=args.seed,
        out_dir=args.out,
    )
