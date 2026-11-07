"""Run reproducible order-book research validation and write reports."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .checks import check_causality, check_data_integrity
from .cv import (
    label_overlap_fraction,
    naive_kfold,
    purged_walk_forward,
)
from .data import load_order_book_csv
from .features import (
    build_dataset,
    build_features,
    build_features_leaky,
    label_tie_fraction,
)
from .metrics import (
    classification_metrics,
    directional_cost_diagnostic,
)
from .models import model_factories
from .reliability import (
    block_bootstrap_ci,
    volatility_regimes,
)
from .simulate import simulate_order_book


def _safe_ci(y, probabilities, metric, block_size, seed):
    """Return a bootstrap confidence interval or an explanatory error."""
    try:
        return block_bootstrap_ci(
            y,
            probabilities,
            metric=metric,
            block_size=block_size,
            n_bootstrap=500,
            confidence=0.95,
            seed=seed,
        )
    except ValueError as exc:
        return {
            "metric": metric,
            "error": str(exc),
        }


def _stability_summary(fold_metrics):
    """Summarize how much a metric changes between validation folds."""
    output = {}

    for metric in (
        "auc",
        "accuracy",
        "balanced_accuracy",
        "brier",
    ):
        values = [
            float(item[metric])
            for item in fold_metrics
            if item.get(metric) is not None
            and np.isfinite(item[metric])
        ]

        if not values:
            output[metric] = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "valid_folds": 0,
            }
            continue

        output[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "valid_folds": len(values),
        }

    return output


def _naive_vs_purged(X, y, factories, purged_auc, n_splits, seed):
    """Same models, same data: shuffled K-fold AUC versus purged AUC."""
    splits = naive_kfold(len(X), n_splits, seed)
    output = {}

    for name, factory in factories.items():
        aucs = []

        for train_idx, test_idx in splits:
            model = factory()
            model.fit(X.iloc[train_idx], y.iloc[train_idx])

            probabilities = model.predict_proba(
                X.iloc[test_idx]
            )[:, 1]

            auc = classification_metrics(
                y.iloc[test_idx].to_numpy(),
                probabilities,
            )["auc"]

            if auc is not None:
                aucs.append(auc)

        naive_auc = float(np.mean(aucs)) if aucs else None
        purged = purged_auc.get(name)

        output[name] = {
            "naive_auc": naive_auc,
            "purged_auc": purged,
            "inflation": (
                naive_auc - purged
                if naive_auc is not None and purged is not None
                else None
            ),
        }

    return output


def _evaluate(
    X,
    y,
    meta,
    splits,
    factories,
    fee_bps,
    slippage_bps,
    horizon,
    seed,
):
    """Evaluate models across folds and volatility regimes."""
    results = {}

    for model_number, (name, factory) in enumerate(factories.items()):
        fold_metrics = []
        costs = []

        all_targets = []
        all_probabilities = []
        all_regimes = []
        regime_thresholds = []

        for fold_number, (train_idx, test_idx) in enumerate(splits):
            model = factory()

            model.fit(
                X.iloc[train_idx],
                y.iloc[train_idx],
            )

            probabilities = model.predict_proba(
                X.iloc[test_idx]
            )[:, 1]

            test_targets = y.iloc[test_idx].to_numpy()

            metrics = classification_metrics(
                test_targets,
                probabilities,
            )
            fold_metrics.append(metrics)

            costs.append(
                directional_cost_diagnostic(
                    probabilities,
                    meta.iloc[test_idx].reset_index(drop=True),
                    fee_bps,
                    slippage_bps,
                )
            )

            # Define regimes using training-fold volatility only.
            train_volatility = X.iloc[train_idx][
                "volatility_20"
            ].to_numpy()

            test_volatility = X.iloc[test_idx][
                "volatility_20"
            ].to_numpy()

            regimes, thresholds = volatility_regimes(
                train_volatility,
                test_volatility,
            )

            regime_thresholds.append({
                "fold": fold_number + 1,
                **thresholds,
            })

            all_targets.extend(test_targets.tolist())
            all_probabilities.extend(
                probabilities.tolist()
            )
            all_regimes.extend(regimes.tolist())

        pooled_y = np.asarray(all_targets, dtype=int)
        pooled_probabilities = np.asarray(
            all_probabilities,
            dtype=float,
        )
        pooled_regimes = np.asarray(all_regimes)

        # Calculate confidence intervals on pooled out-of-sample
        # predictions. Blocks retain local observation ordering.
        confidence_intervals = {
            metric: _safe_ci(
                pooled_y,
                pooled_probabilities,
                metric,
                block_size=max(1, horizon),
                seed=seed + model_number * 100 + metric_number,
            )
            for metric_number, metric in enumerate(
                ("auc", "accuracy", "brier")
            )
        }

        # Evaluate each volatility regime separately.
        regime_results = {}

        for regime in ("low", "normal", "high"):
            mask = pooled_regimes == regime
            count = int(mask.sum())

            if count == 0:
                regime_results[regime] = {
                    "observations": 0,
                    "metrics": None,
                    "note": "No test observations in this regime.",
                }
                continue

            regime_results[regime] = {
                "observations": count,
                "metrics": classification_metrics(
                    pooled_y[mask],
                    pooled_probabilities[mask],
                ),
            }

        numeric_keys = [
            key
            for key, value in fold_metrics[0].items()
            if isinstance(value, (int, float))
            and value is not None
        ]

        averages = {}

        for key in numeric_keys:
            values = [
                item[key]
                for item in fold_metrics
                if item[key] is not None
                and np.isfinite(item[key])
            ]

            averages[key] = (
                round(float(np.mean(values)), 6)
                if values
                else None
            )

        cost_keys = (
            "active_fraction",
            "gross_mean_return_per_row",
            "net_mean_return_per_row",
            "net_sum_return_simple",
        )

        cost_summary = {
            key: round(
                float(np.mean([item[key] for item in costs])),
                8,
            )
            for key in cost_keys
        }
        cost_summary["note"] = costs[0]["note"]

        results[name] = {
            **averages,
            "folds": len(fold_metrics),
            "fold_metrics": fold_metrics,
            "stability": _stability_summary(fold_metrics),
            "confidence_intervals": confidence_intervals,
            "volatility_regimes": regime_results,
            "regime_thresholds_by_fold": regime_thresholds,
            "cost_diagnostic": cost_summary,
        }

    return results


def render_report(report):
    """Render the experiment report as Markdown."""
    lines = [
        "# Leakguard research report",
        "",
        "## Headline: naive K-fold vs purged walk-forward",
        "",
        "Same data, same models. Only the validation method differs.",
        "",
        "| Model | Naive K-fold AUC | Purged walk-forward AUC | Inflation |",
        "|---|---:|---:|---:|",
    ]

    for name, row in report["naive_vs_purged"].items():
        if row["naive_auc"] is None or row["purged_auc"] is None:
            lines.append(f"| {name} | n/a | n/a | n/a |")
            continue

        lines.append(
            f"| {name} | {row['naive_auc']:.3f} | "
            f"{row['purged_auc']:.3f} | "
            f"{row['inflation']:+.3f} |"
        )

    lines += [
        "",
        "Naive K-fold shuffles rows, so training rows sit next to test "
        "rows and share overlapping label windows. Any gap above comes "
        "from the validation method, not from skill. Only the purged "
        "column is an honest estimate.",
        "",
        f"- Data source: **{report['data_source']}**",
        (
            f"- Rows: **{report['input_rows']}** "
            f"(usable: {report['rows_used']})"
        ),
        (
            f"- Horizon: **{report['horizon_ms']} ms** "
            f"(purged over up to {report['purge_rows']} rows)"
            if report["horizon_ms"]
            else f"- Horizon: **{report['horizon']} observations**"
        ),
        (
            f"- Rows where mid-price did not move over the horizon: "
            f"**{report['tie_fraction']:.1%}** "
            + (
                "(dropped, so the label means up vs down)"
                if report["ties_dropped"]
                else "(kept and labelled 0)"
            )
        ),
        f"- SHA-256 data fingerprint: `{report['data_hash']}`",
        (
            f"- Integrity issues: "
            f"**{report['integrity_issues'] or 'none'}**"
        ),
        (
            f"- Causal feature test: "
            f"**{report['causality_clean']}**"
        ),
        (
            "- Deliberately leaky feature detected: **"
            + ("yes" if report["causality_leaky"] else "no")
            + "**"
        ),
        f"- Purged split count: **{report['folds']}**",
        (
            "- Train-label overlap with test: **"
            f"{report['purged_train_test_label_overlap']:.2%}**"
        ),
        "",
        "## Out-of-sample model results",
        "",
        "| Model | AUC | Accuracy | Balanced accuracy | Brier | Calibration error |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name, model in report["models"].items():
        auc = (
            "n/a"
            if model["auc"] is None
            else f"{model['auc']:.4f}"
        )

        lines.append(
            f"| {name} | {auc} | "
            f"{model['accuracy']:.4f} | "
            f"{model['balanced_accuracy']:.4f} | "
            f"{model['brier']:.4f} | "
            f"{model['ece']:.4f} |"
        )

    lines += [
        "",
        "A constant 50/50 forecast scores Brier 0.25. A Brier above 0.25 "
        "means the probabilities are worse than no model at all, even if "
        "AUC looks fine. Calibration error is the average gap between "
        "predicted probability and observed frequency.",
        "",
        "## Confidence intervals",
        "",
        "95% circular moving-block bootstrap intervals:",
        "",
        "| Model | Metric | Estimate | Lower | Upper |",
        "|---|---|---:|---:|---:|",
    ]

    for name, model in report["models"].items():
        for metric, interval in model[
            "confidence_intervals"
        ].items():
            if "error" in interval:
                lines.append(
                    f"| {name} | {metric} | n/a | n/a | n/a |"
                )
                lines.append(
                    f"| | Note | {interval['error']} | | |"
                )
            else:
                lines.append(
                    f"| {name} | {metric} | "
                    f"{interval['estimate']:.4f} | "
                    f"{interval['lower']:.4f} | "
                    f"{interval['upper']:.4f} |"
                )

    lines += [
        "",
        "## Performance by volatility regime",
        "",
        "| Model | Regime | Observations | AUC | Accuracy | Brier |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for name, model in report["models"].items():
        for regime, result in model[
            "volatility_regimes"
        ].items():
            if result["metrics"] is None:
                lines.append(
                    f"| {name} | {regime} | 0 | n/a | n/a | n/a |"
                )
                continue

            metrics = result["metrics"]
            auc = (
                "n/a"
                if metrics["auc"] is None
                else f"{metrics['auc']:.4f}"
            )

            lines.append(
                f"| {name} | {regime} | "
                f"{result['observations']} | "
                f"{auc} | "
                f"{metrics['accuracy']:.4f} | "
                f"{metrics['brier']:.4f} |"
            )

    lines += [
        "",
        "## Stability across validation folds",
        "",
        "Standard deviation measures how much the fold scores vary. "
        "A large value indicates less stable performance.",
        "",
        "| Model | Metric | Mean | Std dev | Minimum | Maximum |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for name, model in report["models"].items():
        for metric, values in model["stability"].items():
            if values["mean"] is None:
                lines.append(
                    f"| {name} | {metric} | n/a | n/a | n/a | n/a |"
                )
                continue

            lines.append(
                f"| {name} | {metric} | "
                f"{values['mean']:.4f} | "
                f"{values['std']:.4f} | "
                f"{values['min']:.4f} | "
                f"{values['max']:.4f} |"
            )

    lines += [
        "",
        "## Cost diagnostic",
        "",
        "This is a simplified diagnostic, not a strategy backtest. "
        "It does not model queue position, fills, market impact, "
        "latency or adverse selection.",
        "",
    ]

    for name, model in report["models"].items():
        cost = model["cost_diagnostic"]
        lines.append(
            f"- **{name}:** "
            f"active={cost['active_fraction']:.1%}, "
            f"net mean/row={cost['net_mean_return_per_row']:.8f}"
        )

    lines += [
        "",
        "## Research interpretation and limitations",
        "",
        "- Synthetic input is for software tests, not evidence of market edge.",
        "- Volatility regimes use training-fold thresholds, not test-derived thresholds.",
        "- Regime metrics can be unreliable when a regime has few observations.",
        "- Bootstrap intervals are approximate and depend on block-size choice.",
        "- The pooled bootstrap does not account for every source of dependence "
        "between folds or repeated experiment selection.",
        "- Fold stability is descriptive; it does not guarantee future performance.",
        "- A CSV input is only as trustworthy as its source, timestamp semantics "
        "and normalization.",
        "- The cost diagnostic is not a realistic execution simulator. "
        "Positions are non-overlapping, one every horizon.",
        "- The naive-vs-purged comparison uses a different train/test shape "
        "(random 80/20 versus expanding window), so the gap is indicative, "
        "not a precise measurement of leakage.",
        "- This report does not establish profitability or readiness for live trading.",
        "",
        "## Reproducibility",
        "",
        f"- Parameters: `{json.dumps(report['params'], sort_keys=True)}`",
        f"- Versions: `{json.dumps(report['versions'], sort_keys=True)}`",
        "",
    ]

    return "\n".join(lines)


def run(
    n=60_000,
    horizon=50,
    seed=7,
    out_dir="results",
    data_path=None,
    n_splits=5,
    initial_train_frac=0.4,
    embargo=None,
    fee_bps=0.0,
    slippage_bps=0.0,
    horizon_ms=None,
    drop_ties=True,
):
    """Run data checks, validation, reliability analysis and reporting."""
    if data_path:
        df = load_order_book_csv(data_path)
        data_source = str(data_path)
    else:
        df = simulate_order_book(n=n, seed=seed)
        data_source = "synthetic_demo_NOT_market_data"

    digest = hashlib.sha256(
        pd.util.hash_pandas_object(
            df,
            index=True,
        ).values.tobytes()
    ).hexdigest()

    integrity = check_data_integrity(df)

    causal = check_causality(build_features, df)
    leaky = check_causality(build_features_leaky, df)

    X, y, meta = build_dataset(
        df,
        horizon,
        return_meta=True,
        ties="drop" if drop_ties else "as_down",
        horizon_ms=horizon_ms,
    )

    tie_fraction = label_tie_fraction(df, horizon, horizon_ms)

    # Purging must cover the longest label window, counted in rows.
    purge_rows = (
        horizon
        if horizon_ms is None
        else int(meta["hold_rows"].max())
    )

    if embargo is None:
        embargo = purge_rows // 5

    if "volatility_20" not in X.columns:
        raise ValueError(
            "The feature dataset must contain volatility_20"
        )

    splits = purged_walk_forward(
        len(X),
        purge_rows,
        n_splits,
        initial_train_frac,
        embargo,
    )

    if not splits:
        raise ValueError(
            "No validation folds could be created. "
            "Use more rows or reduce horizon/splits."
        )

    overlap = max(
        label_overlap_fraction(
            train_idx,
            test_idx,
            purge_rows,
            len(X),
        )
        for train_idx, test_idx in splits
    )

    models = _evaluate(
        X,
        y,
        meta,
        splits,
        model_factories(),
        fee_bps,
        slippage_bps,
        purge_rows,
        seed,
    )

    naive_vs_purged = _naive_vs_purged(
        X,
        y,
        model_factories(),
        {name: result["auc"] for name, result in models.items()},
        n_splits,
        seed,
    )

    report = {
        "data_source": data_source,
        "input_rows": len(df),
        "rows_used": len(X),
        "horizon": horizon,
        "horizon_ms": horizon_ms,
        "purge_rows": purge_rows,
        "tie_fraction": tie_fraction,
        "ties_dropped": drop_ties,
        "naive_vs_purged": naive_vs_purged,
        "data_hash": digest,
        "integrity_issues": integrity,
        "causality_clean": causal,
        "causality_leaky": leaky,
        "folds": len(splits),
        "purged_train_test_label_overlap": overlap,
        "models": models,
        "params": {
            "n": n,
            "horizon": horizon,
            "horizon_ms": horizon_ms,
            "drop_ties": drop_ties,
            "seed": seed,
            "n_splits": n_splits,
            "initial_train_frac": initial_train_frac,
            "embargo": embargo,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "bootstrap_repetitions": 500,
            "confidence_level": 0.95,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    }

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "latest.json").write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
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

    parser.add_argument("--data", help="Normalized order-book CSV")
    parser.add_argument("--n", type=int, default=60_000)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="results")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--embargo", type=int)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument(
        "--horizon-ms",
        type=int,
        help="Predict over this many milliseconds instead of --horizon rows",
    )
    parser.add_argument(
        "--keep-ties",
        action="store_true",
        help="Keep rows with no price move and label them 0 (old behaviour)",
    )

    args = parser.parse_args()

    run(
        n=args.n,
        horizon=args.horizon,
        seed=args.seed,
        out_dir=args.out,
        data_path=args.data,
        n_splits=args.splits,
        embargo=args.embargo,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        horizon_ms=args.horizon_ms,
        drop_ties=not args.keep_ties,
    )
