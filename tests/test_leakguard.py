import numpy as np
from leakguard.checks import (check_causality, check_data_integrity,
                              label_overlap_fraction)
from leakguard.cv import naive_kfold, purged_walk_forward
from leakguard.features import build_dataset, build_features, build_features_leaky
from leakguard.simulate import simulate_order_book

H = 30
df = simulate_order_book(n=4000, seed=1)


def test_simulated_data_is_clean():
    assert check_data_integrity(df) == []


def test_integrity_catches_crossed_book():
    bad = df.copy()
    bad.loc[10, "bid_px_1"] = bad.loc[10, "ask_px_1"] + 1
    assert "crossed book (bid >= ask)" in check_data_integrity(bad)


def test_honest_features_are_causal():
    assert check_causality(build_features, df, n_probes=6) == []


def test_leaky_feature_is_caught():
    assert "LEAK_future_ret" in check_causality(build_features_leaky, df, n_probes=6)


def test_dataset_shapes_match():
    X, y = build_dataset(df, H)
    assert len(X) == len(y) and not X.isna().any().any()


def test_purged_splits_never_overlap_test():
    X, _ = build_dataset(df, H)
    for tr, te in purged_walk_forward(len(X), H, 4, 0.4, embargo=5):
        assert tr.max() + H < te.min()
        assert label_overlap_fraction(tr, te, H, len(X)) == 0.0


def test_naive_kfold_overlaps_heavily():
    X, _ = build_dataset(df, H)
    tr, te = naive_kfold(len(X), 5, 0)[0]
    assert label_overlap_fraction(tr, te, H, len(X)) > 0.9
