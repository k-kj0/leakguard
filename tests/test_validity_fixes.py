"""Tests for tie handling, time horizons, non-overlapping costs and calibration."""

import numpy as np
import pandas as pd
import pytest

from leakguard.experiment import render_report, run
from leakguard.features import build_dataset, label_tie_fraction
from leakguard.metrics import (
    directional_cost_diagnostic,
    expected_calibration_error,
)
from leakguard.simulate import simulate_order_book


def coarse_book(n=3000, tick=1.0):
    """Synthetic book with a coarse tick so many mid-prices do not move."""

    df = simulate_order_book(n=n, seed=5)

    for level in (1, 2, 3):
        df[f"bid_px_{level}"] = np.floor(df[f"bid_px_{level}"] / tick) * tick
        df[f"ask_px_{level}"] = np.ceil(df[f"ask_px_{level}"] / tick) * tick

    return df


def test_ties_are_dropped_by_default():
    df = coarse_book()

    X_drop, y_drop, meta = build_dataset(df, 20, return_meta=True)
    X_keep, y_keep = build_dataset(df, 20, ties="as_down")

    assert label_tie_fraction(df, 20) > 0.05
    assert len(X_drop) < len(X_keep)
    assert (meta["future_mid"] != meta["mid"]).all()
    assert set(y_drop.unique()) == {0, 1}


def test_bad_ties_option_is_rejected():
    with pytest.raises(ValueError):
        build_dataset(simulate_order_book(n=600), 5, ties="maybe")


def test_time_horizon_looks_forward_by_milliseconds():
    df = simulate_order_book(n=3000, seed=2)
    horizon_ms = 2000

    X, y, meta = build_dataset(
        df, 50, return_meta=True, horizon_ms=horizon_ms
    )

    ts = df["ts_ms"].to_numpy()
    start = np.searchsorted(ts, meta["ts_ms"].to_numpy())
    hold = meta["hold_rows"].to_numpy().astype(int)

    assert (ts[start + hold] >= ts[start] + horizon_ms).all()
    assert (ts[start + hold - 1] < ts[start] + horizon_ms).all()
    assert hold.min() >= 1


def test_cost_positions_do_not_overlap():
    n = 100
    meta = pd.DataFrame({
        "mid": np.full(n, 100.0),
        "future_mid": np.full(n, 101.0),
        "spread": np.zeros(n),
        "hold_rows": np.full(n, 10),
    })

    result = directional_cost_diagnostic(np.full(n, 0.9), meta)

    assert result["positions_evaluated"] == 10
    assert result["non_overlap_stride_rows"] == 10
    assert result["net_sum_return_simple"] == pytest.approx(10 * 0.01)


def test_calibration_error_extremes():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=20000)
    y = (rng.uniform(size=20000) < p).astype(int)

    assert expected_calibration_error(y, p) < 0.03

    always_sure = np.full(20000, 0.9)
    coin = rng.integers(0, 2, size=20000)

    assert expected_calibration_error(coin, always_sure) > 0.35


def test_report_leads_with_naive_vs_purged_and_gbm_beats_constant_brier(
    tmp_path,
):
    report = run(n=30000, horizon=50, out_dir=str(tmp_path))
    markdown = render_report(report)

    first_section = markdown.split("\n## ")[1]
    assert first_section.startswith("Headline: naive K-fold vs purged")

    for row in report["naive_vs_purged"].values():
        assert row["naive_auc"] > row["purged_auc"]

    # A constant 50/50 forecast scores 0.25. The regularised boosted
    # model must not be worse than knowing nothing.
    assert report["models"]["hist_gradient_boosting"]["brier"] <= 0.25
