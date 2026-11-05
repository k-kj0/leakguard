"""Out-of-sample metrics and simplified cost diagnostics."""

import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
)


def classification_metrics(
    y_true,
    probabilities,
):

    p = np.asarray(probabilities)
    y = np.asarray(y_true)

    pred = (p >= 0.5).astype(int)

    out = {
        "accuracy": float(
            accuracy_score(y, pred)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),

        "brier": float(
            brier_score_loss(y, p)
        ),
    }

    out["auc"] = (
        float(roc_auc_score(y, p))
        if len(np.unique(y)) == 2
        else None
    )

    return out


def directional_cost_diagnostic(
    probabilities,
    meta,
    fee_bps=0.0,
    slippage_bps=0.0,
    threshold=0.55,
):
    """Toy directional cost diagnostic, not a trading simulator.

    Long when p >= threshold.
    Short when p <= 1 - threshold.
    Otherwise remain flat.

    Assumes immediate execution at mid +/- half-spread.
    Does not model queue position, partial fills, market impact,
    latency or adverse selection.
    """

    p = np.asarray(probabilities)

    direction = np.where(
        p >= threshold,
        1.0,
        np.where(
            p <= 1.0 - threshold,
            -1.0,
            0.0,
        ),
    )

    gross = (
        meta["future_mid"].to_numpy()
        / meta["mid"].to_numpy()
    ) - 1.0

    spread_cost = (
        meta["spread"].to_numpy()
        / meta["mid"].to_numpy()
    )

    fixed_cost = (
        fee_bps + slippage_bps
    ) / 10_000.0

    net = (
        direction * gross
        - (direction != 0) * (
            spread_cost + fixed_cost
        )
    )

    active = direction != 0

    return {
        "threshold": threshold,

        "active_fraction": float(
            active.mean()
        ),

        "gross_mean_return_per_row": float(
            (direction * gross).mean()
        ),

        "net_mean_return_per_row": float(
            net.mean()
        ),

        "net_sum_return_simple": float(
            net.sum()
        ),

        "note": (
            "Diagnostic only; not a realistic execution/backtest model."
        ),
    }
