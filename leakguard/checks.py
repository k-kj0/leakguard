"""Leakage and data-sanity checks. Each returns something you can assert on."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def check_data_integrity(df: pd.DataFrame) -> list:
    issues = []
    if not df["ts_ms"].is_monotonic_increasing:
        issues.append("timestamps are not sorted")
    if df["ts_ms"].duplicated().any():
        issues.append("duplicate timestamps")
    if df.isna().any().any():
        issues.append("NaN values present")
    if (df["bid_px_1"] >= df["ask_px_1"]).any():
        issues.append("crossed book (bid >= ask)")
    return issues


def check_causality(feature_fn, df: pd.DataFrame, n_probes: int = 15,
                    seed: int = 0, min_row: int = 200) -> list:
    """Truncation test. Pretend we are standing at row k: delete everything after k,
    recompute features, and compare the last row with the full-data value.
    A causal feature cannot change. Returns names of features that did."""
    df = df.reset_index(drop=True)
    full = feature_fn(df)
    rng = np.random.default_rng(seed)
    probes = rng.choice(np.arange(min_row, len(df) - 1), size=n_probes, replace=False)
    bad = set()
    for k in probes:
        part = feature_fn(df.iloc[: k + 1].reset_index(drop=True)).iloc[-1]
        ref = full.iloc[k]
        for col in full.columns:
            if not np.isclose(part[col], ref[col], rtol=1e-9, atol=1e-12, equal_nan=True):
                bad.add(col)
    return sorted(bad)


def label_overlap_fraction(train_idx, test_idx, horizon: int, n: int) -> float:
    """Share of training rows whose label window [i, i+horizon] overlaps any test label window."""
    diff = np.zeros(n + horizon + 2, dtype=int)
    np.add.at(diff, test_idx, 1)
    np.add.at(diff, test_idx + horizon + 1, -1)
    covered = (np.cumsum(diff) > 0).astype(int)
    cs = np.concatenate([[0], np.cumsum(covered)])
    train_idx = np.asarray(train_idx)
    overlap = (cs[train_idx + horizon + 1] - cs[train_idx]) > 0
    return float(overlap.mean())


def shuffled_label_auc(model_factory, X, y, splits, seed: int = 0) -> float:
    """Sanity check: with labels shuffled there is nothing to learn, so AUC must be ~0.5.
    If it is not, the pipeline is leaking or buggy."""
    y_shuf = pd.Series(np.random.default_rng(seed).permutation(y.values))
    aucs = []
    for tr, te in splits:
        m = model_factory()
        m.fit(X.iloc[tr], y_shuf.iloc[tr])
        aucs.append(roc_auc_score(y_shuf.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(aucs))
