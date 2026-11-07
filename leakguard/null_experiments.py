
"""Null experiments using shuffled labels and chronological validation."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from .cv import purged_walk_forward
from .data import simulate_order_book
from .features import build_dataset
from .models import model_factories


def run_null_experiments(
    n=3000,
    horizon=10,
    repetitions=20,
    seed=7,
    n_splits=3,
    out_dir="results/null_experiments",
):
    """
    Test a baseline model after independently shuffling the labels.

    The shuffled labels destroy the original row-to-label relationship.
    This is an IID label-shuffle negative control, not a realistic market
    simulation or a proof that all forms of leakage are absent.
    """
    if n < 500:
        raise ValueError("Use at least 500 synthetic rows.")

    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    if repetitions < 2:
        raise ValueError("Use at least 2 repetitions.")

    if n_splits < 2:
        raise ValueError("Use at least 2 validation splits.")

    df = simulate_order_book(n=n, seed=seed)
    X, y = build_dataset(df, horizon=horizon)

    splits = purged_walk_forward(
        n=len(X),
        horizon=horizon,
        n_splits=n_splits,
        initial_train_frac=0.5,
        embargo=max(1, horizon // 5),
    )

    if len(splits) < 2:
        raise ValueError(
            "Not enough observations for the requested validation splits."
        )

    rng = np.random.default_rng(seed)
    factory = model_factories()["logistic_regression"]
    results = []

    original_labels = y.to_numpy()

    for repetition in range(repetitions):
        shuffled_labels = rng.permutation(original_labels)

        all_targets = []
        all_probabilities = []

        for train_idx, test_idx in splits:
            model = factory()

            train_y = shuffled_labels[train_idx]
            test_y = shuffled_labels[test_idx]

            if np.unique(train_y).size < 2:
                raise ValueError(
                    "A shuffled training fold contains only one class."
                )

            model.fit(
                X.iloc[train_idx],
                train_y,
            )

            probabilities = model.predict_proba(
                X.iloc[test_idx]
            )[:, 1]

            all_targets.extend(test_y.tolist())
            all_probabilities.extend(probabilities.tolist())

        targets = np.asarray(all_targets, dtype=int)
        probabilities = np.asarray(all_probabilities, dtype=float)

        auc = float(roc_auc_score(targets, probabilities))
        accuracy = float(
            accuracy_score(targets, probabilities >= 0.5)
        )

        results.append({
            "repetition": repetition + 1,
            "auc": auc,
            "accuracy": accuracy,
            "test_observations": int(len(targets)),
        })

    auc_values = np.asarray(
        [item["auc"] for item in results],
        dtype=float,
    )
    accuracy_values = np.asarray(
        [item["accuracy"] for item in results],
        dtype=float,
    )

    summary = {
        "repetitions": repetitions,
        "mean_auc": float(np.mean(auc_values)),
        "std_auc": float(np.std(auc_values, ddof=1)),
        "median_auc": float(np.median(auc_values)),
        "auc_2_5_percentile": float(np.quantile(auc_values, 0.025)),
        "auc_97_5_percentile": float(np.quantile(auc_values, 0.975)),
        "mean_accuracy": float(np.mean(accuracy_values)),
        "fraction_auc_at_least_0_60": float(
            np.mean(auc_values >= 0.60)
        ),
        "expected_auc_reference": 0.5,
    }

    report = {
        "experiment": "IID shuffled-label null experiment",
        "seed": seed,
        "input_rows": len(df),
        "usable_rows": len(X),
        "horizon_observations": horizon,
        "validation_splits": len(splits),
        "model": "logistic_regression",
        "null_method": "independent permutation of labels",
        "summary": summary,
        "repetitions": results,
        "limitations": [
            "The order-book input is synthetic.",
            "Labels are independently shuffled, destroying their original row association.",
            "This is an IID label-shuffle control, not a block-preserving time-series null.",
            "The experiment tests one baseline model and one feature set.",
            "It does not rule out every form of data leakage or selection bias.",
            "AUC values near 0.5 are consistent with no ranking signal under this null.",
            "This is not evidence of trading profitability.",
        ],
    }

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "latest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# Leakguard Null Experiment Report",
        "",
        f"- Seed: **{seed}**",
        f"- Synthetic input rows: **{len(df)}**",
        f"- Usable feature rows: **{len(X)}**",
        f"- Prediction horizon: **{horizon} observations**",
        f"- Model: **logistic regression**",
        f"- Independent label-shuffle repetitions: **{repetitions}**",
        f"- Validation folds: **{len(splits)}**",
        "",
        "## Null results",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Mean AUC | {summary['mean_auc']:.4f} |",
        f"| AUC standard deviation | {summary['std_auc']:.4f} |",
        f"| Median AUC | {summary['median_auc']:.4f} |",
        f"| AUC 2.5th percentile | {summary['auc_2_5_percentile']:.4f} |",
        f"| AUC 97.5th percentile | {summary['auc_97_5_percentile']:.4f} |",
        f"| Mean accuracy | {summary['mean_accuracy']:.4f} |",
        f"| Fraction of runs with AUC >= 0.60 | {summary['fraction_auc_at_least_0_60']:.1%} |",
        "",
        "The reference AUC under a no-ranking-signal null is 0.5.",
        "The reported percentiles describe this simulation's repeated outcomes; "
        "they are not a confidence interval for live market performance.",
        "",
        "## Individual repetitions",
        "",
        "| Repetition | AUC | Accuracy | Test observations |",
        "|---:|---:|---:|---:|",
    ]

    for item in results:
        lines.append(
            f"| {item['repetition']} | {item['auc']:.4f} | "
            f"{item['accuracy']:.4f} | {item['test_observations']} |"
        )

    lines += [
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
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--out", default="results/null_experiments")

    args = parser.parse_args()

    run_null_experiments(
        n=args.n,
        horizon=args.horizon,
        repetitions=args.repetitions,
        seed=args.seed,
        n_splits=args.splits,
        out_dir=args.out,
    )
