"""Reproduce the core result of Cont, Kukanov and Stoikov (2014).

Paper: "The Price Impact of Order Book Events", Journal of Financial
Econometrics 12(1), 47-88.

Their finding: over short intervals, mid-price changes are roughly linear
in order flow imbalance (OFI) at the best bid and ask, with an average
R-squared of about 65% across stocks.

This module computes level-1 OFI, regresses the mid-price change on it, and
then asks the question the paper does not: does OFI in one interval predict
the price change in the NEXT interval? That is the tradable version, and it
is where the contemporaneous result and a usable signal part ways.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_order_book_csv
from .features import mid_price
from .simulate import simulate_order_book

PAPER_AVERAGE_R2 = 0.65


def order_flow_imbalance_events(df: pd.DataFrame) -> np.ndarray:
    """Level-1 order flow imbalance contribution of each row (paper eq. 3).

    e_n = 1{bid_n >= bid_prev} * bid_size_n
        - 1{bid_n <= bid_prev} * bid_size_prev
        - 1{ask_n <= ask_prev} * ask_size_n
        + 1{ask_n >= ask_prev} * ask_size_prev

    The first row has no previous row, so its contribution is 0.
    """

    bid = df["bid_px_1"].to_numpy(dtype=float)
    ask = df["ask_px_1"].to_numpy(dtype=float)
    bid_size = df["bid_sz_1"].to_numpy(dtype=float)
    ask_size = df["ask_sz_1"].to_numpy(dtype=float)

    events = np.zeros(len(df))

    if len(df) < 2:
        return events

    bid_now, bid_prev = bid[1:], bid[:-1]
    ask_now, ask_prev = ask[1:], ask[:-1]

    events[1:] = (
        (bid_now >= bid_prev) * bid_size[1:]
        - (bid_now <= bid_prev) * bid_size[:-1]
        - (ask_now <= ask_prev) * ask_size[1:]
        + (ask_now >= ask_prev) * ask_size[:-1]
    )

    return events


def _ols(x: np.ndarray, y: np.ndarray) -> dict:
    """Simple regression y = a + b x, returning slope, intercept, R-squared."""

    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"slope": None, "intercept": None, "r2": None}

    slope, intercept = np.polyfit(x, y, 1)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
    }


def ofi_regression(df: pd.DataFrame, bin_ms: int = 10_000) -> dict:
    """Regress mid-price change on OFI inside fixed time bins."""

    if bin_ms < 1:
        raise ValueError("bin_ms must be >= 1")

    df = df.reset_index(drop=True)
    mid = mid_price(df).to_numpy()
    depth = (df["bid_sz_1"] + df["ask_sz_1"]).to_numpy(dtype=float) / 2.0

    frame = pd.DataFrame({
        "bin": (df["ts_ms"] - df["ts_ms"].iloc[0]) // bin_ms,
        "ofi": order_flow_imbalance_events(df),
        "mid": mid,
        "depth": depth,
    })

    grouped = frame.groupby("bin")

    bins = pd.DataFrame({
        "ofi": grouped["ofi"].sum(),
        "end_mid": grouped["mid"].last(),
        "depth": grouped["depth"].mean(),
    })

    # Price change over a bin runs from the previous bin's last mid
    # to this bin's last mid, matching the events summed in the bin.
    bins["d_mid"] = bins["end_mid"].diff()
    bins = bins.dropna(subset=["d_mid"])

    bins["ofi_scaled"] = bins["ofi"] / bins["depth"]

    contemporaneous = _ols(
        bins["ofi"].to_numpy(),
        bins["d_mid"].to_numpy(),
    )

    contemporaneous_scaled = _ols(
        bins["ofi_scaled"].to_numpy(),
        bins["d_mid"].to_numpy(),
    )

    # Predictive version: OFI in bin b versus price change in bin b+1.
    predictive = _ols(
        bins["ofi"].to_numpy()[:-1],
        bins["d_mid"].to_numpy()[1:],
    )

    return {
        "bin_ms": bin_ms,
        "bins_used": int(len(bins)),
        "contemporaneous": contemporaneous,
        "contemporaneous_depth_scaled": contemporaneous_scaled,
        "predictive_next_bin": predictive,
        "paper_average_r2": PAPER_AVERAGE_R2,
    }


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3g}"


def render_markdown(result: dict, data_source: str) -> str:
    contemporaneous = result["contemporaneous"]
    scaled = result["contemporaneous_depth_scaled"]
    predictive = result["predictive_next_bin"]

    return "\n".join([
        "# Reproduction: Cont, Kukanov and Stoikov (2014)",
        "",
        "Paper: *The Price Impact of Order Book Events*, Journal of "
        "Financial Econometrics 12(1), 47-88.",
        "",
        f"- Data source: **{data_source}**",
        f"- Bin length: **{result['bin_ms'] / 1000:g} s**, "
        f"bins used: **{result['bins_used']}**",
        "",
        "| Regression | R-squared | Slope |",
        "|---|---:|---:|",
        f"| Price change on OFI, same bin (this repo) | "
        f"{_fmt(contemporaneous['r2'])} | {_fmt(contemporaneous['slope'])} |",
        f"| Same, OFI divided by depth | "
        f"{_fmt(scaled['r2'])} | {_fmt(scaled['slope'])} |",
        f"| Price change in NEXT bin on OFI (predictive) | "
        f"{_fmt(predictive['r2'])} | {_fmt(predictive['slope'])} |",
        f"| Paper, average across stocks | "
        f"~{result['paper_average_r2']:.2f} | n/a |",
        "",
        "## Where this differs from the paper",
        "",
        "- The paper uses event-by-event NYSE quote data for U.S. stocks. "
        "This repo uses fixed-grid snapshots of one crypto product, so "
        "events that happen between two snapshots are missed.",
        "- The paper averages R-squared across many stocks. This is one "
        "product over one capture window.",
        "- The paper's R-squared is contemporaneous: OFI and the price "
        "change are measured over the same interval. That explains "
        "prices after the fact. It cannot be traded. The predictive row "
        "is the test that matters for a forecasting model.",
        "- Only level-1 OFI is used here, as in the paper's baseline.",
        "- A capture of a few minutes gives few bins, so every R-squared "
        "here has wide uncertainty.",
        "",
        "Synthetic input only tests that the code runs. It says nothing "
        "about markets.",
        "",
    ])


def run(
    data_path=None,
    bin_ms: int = 10_000,
    out_dir: str = "results",
    n: int = 60_000,
    seed: int = 7,
) -> dict:
    if data_path:
        df = load_order_book_csv(data_path)
        data_source = str(data_path)
    else:
        df = simulate_order_book(n=n, seed=seed)
        data_source = "synthetic_demo_NOT_market_data"

    result = ofi_regression(df, bin_ms=bin_ms)
    markdown = render_markdown(result, data_source)

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "ofi_reproduction.md").write_text(markdown, encoding="utf-8")

    print(markdown)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Normalized order-book CSV")
    parser.add_argument("--bin-ms", type=int, default=10_000)
    parser.add_argument("--out", default="results")
    parser.add_argument("--n", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    run(
        data_path=args.data,
        bin_ms=args.bin_ms,
        out_dir=args.out,
        n=args.n,
        seed=args.seed,
    )
