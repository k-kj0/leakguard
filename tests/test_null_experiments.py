
import json

import pytest

from leakguard.null_experiments import run_null_experiments


def test_null_experiment_creates_reports(tmp_path):
    report = run_null_experiments(
        n=1200,
        horizon=5,
        repetitions=3,
        seed=7,
        n_splits=2,
        out_dir=tmp_path,
    )

    assert report["experiment"] == "IID shuffled-label null experiment"
    assert len(report["repetitions"]) == 3

    assert (tmp_path / "latest.json").is_file()
    assert (tmp_path / "latest.md").is_file()


def test_null_experiment_results_are_finite(tmp_path):
    report = run_null_experiments(
        n=1200,
        horizon=5,
        repetitions=3,
        seed=11,
        n_splits=2,
        out_dir=tmp_path,
    )

    summary = report["summary"]

    assert 0.0 <= summary["mean_auc"] <= 1.0
    assert 0.0 <= summary["mean_accuracy"] <= 1.0
    assert summary["std_auc"] >= 0.0
    assert 0.0 <= summary["fraction_auc_at_least_0_60"] <= 1.0


def test_null_experiment_is_reproducible(tmp_path):
    first = run_null_experiments(
        n=1200,
        horizon=5,
        repetitions=3,
        seed=42,
        n_splits=2,
        out_dir=tmp_path / "first",
    )

    second = run_null_experiments(
        n=1200,
        horizon=5,
        repetitions=3,
        seed=42,
        n_splits=2,
        out_dir=tmp_path / "second",
    )

    assert first == second


def test_null_experiment_rejects_invalid_parameters(tmp_path):
    with pytest.raises(ValueError, match="at least 500"):
        run_null_experiments(n=499, out_dir=tmp_path)

    with pytest.raises(ValueError, match="at least 2"):
        run_null_experiments(
            n=1200,
            repetitions=1,
            out_dir=tmp_path,
        )

    with pytest.raises(ValueError, match="at least 1"):
        run_null_experiments(
            n=1200,
            horizon=0,
            out_dir=tmp_path,
        )


def test_null_experiment_json_is_valid(tmp_path):
    run_null_experiments(
        n=1200,
        horizon=5,
        repetitions=3,
        seed=7,
        n_splits=2,
        out_dir=tmp_path,
    )

    report = json.loads(
        (tmp_path / "latest.json").read_text(encoding="utf-8")
    )

    assert report["null_method"] == "independent permutation of labels"
    assert report["summary"]["expected_auc_reference"] == 0.5
    assert len(report["repetitions"]) == 3
