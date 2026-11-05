"""Run reproducible order-book research validation and write reports."""

import argparse
import hashlib
import json
import platform

from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .checks import (
    check_causality,
    check_data_integrity,
)

from .cv import (
    purged_walk_forward,
    label_overlap_fraction,
)

from .data import load_order_book_csv

from .features import (
    build_dataset,
    build_features,
    build_features_leaky,
)

from .metrics import (
    classification_metrics,
    directional_cost_diagnostic,
)

from .models import model_factories
from .simulate import simulate_order_book


def _evaluate(
    X,
    y,
    meta,
    splits,
    factories,
    fee_bps,
    slippage_bps,
):

    results = {}

    for name, factory in factories.items():

        fold_metrics = []
        costs = []

        for train_idx, test_idx in splits:

            model = factory()

            model.fit(
                X.iloc[train_idx],
                y.iloc[train_idx],
            )

            prob = model.predict_proba(
                X.iloc[test_idx]
            )[:, 1]

            fold_metrics.append(
                classification_metrics(
                    y.iloc[test_idx],
                    prob,
                )
            )

            costs.append(
                directional_cost_diagnostic(
                    prob,
                    meta.iloc[test_idx].reset_index(drop=True),
                    fee_bps,
                    slippage_bps,
                )
            )

        keys = [
            k for k in fold_metrics[0]
            if isinstance(
                fold_metrics[0][k],
                (int, float),
            )
        ]

        results[name] = {
            k: round(
                float(
                    np.nanmean([
                        m[k]
                        for m in fold_metrics
                        if m[k] is not None
                    ])
                ),
                6,
            )
            for k in keys
        }

        results[name]["folds"] = len(fold_metrics)

        results[name]["cost_diagnostic"] = {

            k: round(
                float(
                    np.mean([
                        c[k]
                        for c in costs
                    ])
                ),
                8,
            )

            for k in (
                "active_fraction",
                "gross_mean_return_per_row",
                "net_mean_return_per_row",
                "net_sum_return_simple",
            )
        }

        results[name]["cost_diagnostic"]["note"] = (
            costs[0]["note"]
        )

    return results


def render_report(r):

    lines = [
        "# Leakguard research report",
        "",
        f"- Data source: **{r['data_source']}**",
        f"- Rows: **{r['input_rows']}** (usable: {r['rows_used']})",
        f"- Horizon: **{r['horizon']} observations**",
        f"- SHA-256 data fingerprint: `{r['data_hash']}`",
        f"- Integrity issues: **{r['integrity_issues'] or 'none'}**",
        f"- Causal feature test: **{r['causality_clean'] or 'passed'}**",
        (
            "- Deliberately leaky feature detected: **"
            + ("yes" if r["causality_leaky"] else "no")
            + "**"
        ),
        f"- Purged split count: **{r['folds']}**",
        (
            "- Train-label overlap with test: **"
            f"{r['purged_train_test_label_overlap']:.2%}**"
        ),
        "",
        "## Out-of-sample model results",
        "",
        "| Model | AUC | Accuracy | Balanced accuracy | Brier |",
        "|---|---:|---:|---:|---:|",
    ]

    for name, m in r["models"].items():

        auc = (
            "n/a"
            if m["auc"] is None
            else f"{m['auc']:.4f}"
        )

        lines.append(
            f"| {name} | {auc} | "
            f"{m['accuracy']:.4f} | "
            f"{m['balanced_accuracy']:.4f} | "
            f"{m['brier']:.4f} |"
        )

    lines += [
        "",
        "## Cost diagnostic",
        "",
        (
            "This is a simplified diagnostic, not a strategy backtest. "
            "It does not model queue position, fills, market impact, "
            "latency or adverse selection."
        ),
        "",
    ]

    for name, m in r["models"].items():

        c = m["cost_diagnostic"]

        lines.append(
            f"- **{name}:** "
            f"active={c['active_fraction']:.1%}, "
            f"net mean/row={c['net_mean_return_per_row']:.8f}"
        )

    lines += [
        "",
        "## Reproducibility",
        "",
        f"- Parameters: `{json.dumps(r['params'], sort_keys=True)}`",
        f"- Versions: `{json.dumps(r['versions'], sort_keys=True)}`",
        "",
        "## Interpretation limits",
        "",
        "- Synthetic input is for software tests only; it is not evidence of market edge.",
        "- A CSV input is only as trustworthy as its source, timestamp semantics and normalization.",
        "- This report does not establish profitability or readiness for live trading.",
        "- Compare models across all folds and inspect regime stability before drawing conclusions.",
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
):

    if data_path:

        df = load_order_book_csv(data_path)
        data_source = str(data_path)

    else:

        df = simulate_order_book(
            n=n,
            seed=seed,
        )

        data_source = "synthetic_demo_NOT_market_data"

    digest = hashlib.sha256(
        pd.util.hash_pandas_object(
            df,
            index=True,
        ).values.tobytes()
    ).hexdigest()

    embargo = (
        horizon // 5
        if embargo is None
        else embargo
    )

    integrity = check_data_integrity(df)

    causal = check_causality(
        build_features,
        df,
    )

    leaky = check_causality(
        build_features_leaky,
        df,
    )

    X, y, meta = build_dataset(
        df,
        horizon,
        return_meta=True,
    )

    splits = purged_walk_forward(
        len(X),
        horizon,
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
            tr,
            te,
            horizon,
            len(X),
        )
        for tr, te in splits
    )

    models = _evaluate(
        X,
        y,
        meta,
        splits,
        model_factories(),
        fee_bps,
        slippage_bps,
    )

    report = {

        "data_source": data_source,

        "input_rows": len(df),

        "rows_used": len(X),

        "horizon": horizon,

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
            "seed": seed,
            "n_splits": n_splits,
            "initial_train_frac": initial_train_frac,
            "embargo": embargo,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
        },

        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    }

    out = Path(out_dir)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    (out / "latest.json").write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        )
    )

    (out / "latest.md").write_text(
        render_report(report)
    )

    print(render_report(report))

    return report


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--data",
        help="Normalized order-book CSV",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=60_000,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--out",
        default="results",
    )

    parser.add_argument(
        "--splits",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--embargo",
        type=int,
    )

    parser.add_argument(
        "--fee-bps",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=0.0,
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
    )
