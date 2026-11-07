"""Tests for the order-flow-imbalance reproduction."""

import numpy as np
import pandas as pd
import pytest

from leakguard.ofi_reproduction import (
    _ols,
    ofi_regression,
    order_flow_imbalance_events,
)
from leakguard.simulate import simulate_order_book


def test_ofi_events_match_hand_calculation():
    df = pd.DataFrame({
        "bid_px_1": [100.0, 100.0, 101.0, 100.5],
        "ask_px_1": [101.0, 101.0, 102.0, 101.5],
        "bid_sz_1": [10.0, 15.0, 4.0, 6.0],
        "ask_sz_1": [10.0, 10.0, 7.0, 3.0],
    })

    # row 1: bid size +5, ask unchanged           -> 5
    # row 2: bid up (+4), ask up (+10 prev size)   -> 14
    # row 3: bid down (-4 prev), ask down (-3)     -> -7
    assert order_flow_imbalance_events(df).tolist() == [0.0, 5.0, 14.0, -7.0]


def test_ols_recovers_a_known_line():
    x = np.arange(10, dtype=float)
    result = _ols(x, 2.0 * x + 1.0)

    assert result["slope"] == pytest.approx(2.0)
    assert result["intercept"] == pytest.approx(1.0)
    assert result["r2"] == pytest.approx(1.0)


def test_ols_handles_a_flat_input():
    assert _ols(np.ones(10), np.arange(10.0))["r2"] is None


def test_regression_runs_and_reports_all_three_rows():
    df = simulate_order_book(n=20000, seed=1)
    result = ofi_regression(df, bin_ms=10_000)

    assert result["bins_used"] > 50
    assert 0.0 <= result["contemporaneous"]["r2"] <= 1.0
    assert 0.0 <= result["predictive_next_bin"]["r2"] <= 1.0
    assert result["paper_average_r2"] == 0.65


def test_bad_bin_length_is_rejected():
    with pytest.raises(ValueError):
        ofi_regression(simulate_order_book(n=600), bin_ms=0)
