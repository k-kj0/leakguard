
"""Simulate the winner's curse from selecting the best random candidate."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def run_selection_bias(
    n_observations=1000,
    n_trials=50,
    repetitions=200,
    seed=7,
    selection_threshold=0.60,
    out_dir="results/selection_bias",
):
    """
    Compare one random candidate with the best of many random candidates.

    Every candidate score is independent of the binary labels. Selecting
    the maximum AUC on the same evaluation sample illustrates selection
    optimism under a null with no predictive signal.
    """
    if n_observations < 100:
        raise ValueError("Use at least 100 observations.")

    if n_trials < 2:
        raise ValueError("Use at least 2 candidate trials.")

    if repetitions < 20:
        raise ValueError("Use at least 20 repetitions.")

    if not 0.5 < selection_threshold < 1.0:
        raise ValueError(
            "selection_threshold must be between 0.5 and 1.0."
        )

    rng = np.random.default_rng(seed)
    results = []

    for repetition in range(repetitions):
        labels = rng.integers(
            0,
            2,
            size=n_observations,
        )

        candidate_aucs = []

        for _ in range(n_trials):
            random_scores = rng.normal(
                loc=0.0,
                scale=1.0,
                size=n_observations,
            )

            candidate_aucs.append(
                float(roc_auc_score(labels, random_scores))
            )

        single_auc = candidate_aucs[0]
        selected_auc = max(candidate_aucs)

        results.append({
            "repetition": repetition + 1,
            "single_candidate_auc": single_auc,
            "best_candidate_auc": selected_auc,
        })

    single_values = np.asarray(
        [item["single_candidate_auc"] for item in results],
        dtype=float,
    )

    selected_values = np.asarray(
        [item["best_candidate_auc"] for item in results],
        dtype=float,
    )

    summary = {
        "repetitions": repetitions,
        "candidate_trials_per_repetition": n_trials,
        "mean_single_candidate_auc": float(np.mean(single_values)),
        "mean_selected_candidate_auc": float(
            np.mean(selected_values)
        ),
        "mean_selection_uplift": float(
            np.mean(selected_values - single_values)
        ),
        "median_selected_candidate_auc": float(
            np.median(selected_values)
        ),
        "selected_auc_95th_percentile": float(
            np.quantile(selected_values, 0.95)
        ),
        "fraction_selected_auc_at_least_threshold": float(
            np.mean(selected_values >= selection_threshold)
        ),
        "selection_threshold": selection_threshold,
        "null_reference_auc": 0.5,
    }

    report = {
        "experiment": "Random-candidate selection bias simulation",
        "seed": seed,
        "observations_per_repetition": n_observations,
        "candidate_trials_per_repetition": n_trials,
        "selection_rule": "choose the candidate with the highest AUC",
        "evaluation_reuse": (
            "The same labels are used to score all candidates and select "
            "the winner, illustrating selection on the evaluation sample."
        ),
        "summary": summary,
        "repetitions": results,
        "limitations": [
            "All candidate scores are random and independent of the labels.",
            "This is a controlled null simulation, not a model benchmark.",
            "The selected candidate is evaluated on the same sample used for selection.",
            "The experiment illustrates selection optimism, not a correction for it.",
            "The results depend on observation count, candidate count and threshold.",
            "This does not estimate the number of trials in any real research project.",
        ],
    }

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "latest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# Leakguard Selection-Bias Simulation",
        "",
        f"- Seed: **{seed}**",
        f"- Observations per repetition: **{n_observations}**",
        f"- Random candidate trials per repetition: **{n_trials}**",
        f"- Repetitions: **{repetitions}**",
        "- Candidate selection: **highest AUC on the same evaluation sample**",
        "",
        "## Results",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Mean AUC of first random candidate | {summary['mean_single_candidate_auc']:.4f} |",
        f"| Mean AUC of selected best candidate | {summary['mean_selected_candidate_auc']:.4f} |",
        f"| Mean selection uplift | {summary['mean_selection_uplift']:.4f} |",
        f"| Median selected AUC | {summary['median_selected_candidate_auc']:.4f} |",
        f"| 95th percentile selected AUC | {summary['selected_auc_95th_percentile']:.4f} |",
        f"| Fraction selected AUC >= {selection_threshold:.2f} | "
        f"{summary['fraction_selected_auc_at_least_threshold']:.1%} |",
        "",
        "The candidates contain no predictive signal. A selected score can "
        "look better simply because the best of many noisy results was chosen.",
        "",
        "## Interpretation",
        "",
        "- The first random candidate is a reference for a single unselected trial.",
        "- The selected candidate is the maximum across all candidate trials.",
        "- The difference illustrates the winner's curse from repeated evaluation.",
        "- A separate untouched test set or a properly designed nested evaluation "
        "is needed to assess a selected research candidate.",
        "",
        "## Limitations",
        "",
    ]

    lines.extend(f"- {item}" for item in report["limitations"])

    markdown = "\n".join(lines) + "\n"

    (output / "latest.md").write_text(
        markdown,
        encoding="utf-8",
    )

    print(markdown)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--out", default="results/selection_bias")

    args = parser.parse_args()

    run_selection_bias(
        n_observations=args.observations,
        n_trials=args.trials,
        repetitions=args.repetitions,
        seed=args.seed,
        selection_threshold=args.threshold,
        out_dir=args.out,
    )
