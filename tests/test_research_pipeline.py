import numpy as np
import pandas as pd
import pytest

from leakguard.simulate import simulate_order_book

from leakguard.features import (
    build_dataset,
    build_features,
    build_features_leaky,
)

from leakguard.checks import (
    check_causality,
    check_data_integrity,
)

from leakguard.cv import (
    purged_walk_forward,
    label_overlap_fraction,
)

from leakguard.data import load_order_book_csv


@pytest.fixture(scope="module")
def book():

    return simulate_order_book(
        6000,
        seed=11,
    )


def test_simulator_is_integrity_clean(book):

    assert check_data_integrity(book) == []


def test_causality_detector_passes_clean_features(book):

    assert check_causality(
        build_features,
        book,
        n_probes=5,
    ) == []


def test_causality_detector_catches_future_feature(book):

    detected = check_causality(
        build_features_leaky,
        book,
        n_probes=5,
    )

    assert "LEAK_future_ret" in detected


def test_purged_splits_have_zero_label_overlap(book):

    X, _ = build_dataset(
        book,
        horizon=30,
    )

    splits = purged_walk_forward(
        len(X),
        30,
        n_splits=4,
        embargo=6,
    )

    assert len(splits) == 4

    for train, test in splits:

        assert train.max() + 30 < test.min()

        assert label_overlap_fraction(
            train,
            test,
            30,
            len(X),
        ) == 0.0


def test_dataset_has_aligned_labels_and_features(book):

    X, y, meta = build_dataset(
        book,
        25,
        return_meta=True,
    )

    assert len(X) == len(y) == len(meta)

    assert not X.isna().any().any()

    assert set(y.unique()).issubset({0, 1})


def test_csv_loader_rejects_unsorted_timestamps(
    tmp_path,
    book,
):

    bad = book.iloc[::-1]

    path = tmp_path / "bad.csv"

    bad.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="sorted",
    ):

        load_order_book_csv(path)


def test_csv_loader_accepts_normalized_csv(
    tmp_path,
    book,
):

    path = tmp_path / "book.csv"

    book.to_csv(
        path,
        index=False,
    )

    loaded = load_order_book_csv(path)

    assert len(loaded) == len(book)
