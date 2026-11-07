
import json

import pytest

from leakguard.null_experiments import run_null_experiments


def test_null_experiment_creates_reports(tmp_path):
    report = run_null_experiments(
        n_observations=300,
        repetitions=100,
        seed=7,
        out_dir=tmp_path,
    )

    assert report["version"] == 1
    assert report["experiment"] == "independent_random_label_null"

    assert (tmp_path / "latest.json").is_file()
    assert (tmp_path / "latest.md").is_file()


def test_null_experiment_auc_is_near_chance(tmp_path):
    report = run_null_experiments(
        n_observations=300,
        repetitions=200,
        seed=19,
        out_dir=tmp_path,
    )

    summary = report["summary"]

    assert summary["null_reference_auc"] == pytest.approx(0.5)
    assert 0.45 < summary["mean_auc"] < 0.55
    assert 0.0 < summary["std_auc"] < 0.15
    assert 0.0 <= summary["fraction_auc_ge_060"] <= 1.0


def test_null_experiment_contains_both_sides_of_chance(tmp_path):
    report = run_null_experiments(
        n_observations=300,
        repetitions=200,
        seed=7,
        out_dir=tmp_path,
    )

    summary = report["summary"]

    assert summary["trials_below_050"] > 0
    assert summary["trials_above_050"] > 0

    total = (
        summary["trials_below_050"]
        + summary["trials_equal_050"]
        + summary["trials_above_050"]
    )

    assert total == 200


def test_null_experiment_is_reproducible(tmp_path):
    first = run_null_experiments(
        n_observations=300,
        repetitions=100,
        seed=42,
        out_dir=tmp_path / "first",
    )

    second = run_null_experiments(
        n_observations=300,
        repetitions=100,
        seed=42,
        out_dir=tmp_path / "second",
    )

    assert first == second


def test_null_experiment_json_matches_returned_report(tmp_path):
    report = run_null_experiments(
        n_observations=300,
        repetitions=100,
        seed=7,
        out_dir=tmp_path,
    )

    saved = json.loads(
        (tmp_path / "latest.json").read_text(encoding="utf-8")
    )

    assert saved == report


def test_null_experiment_markdown_contains_interpretation(tmp_path):
    run_null_experiments(
        n_observations=300,
        repetitions=100,
        seed=7,
        out_dir=tmp_path,
    )

    content = (
        tmp_path / "latest.md"
    ).read_text(encoding="utf-8")

    assert "Null AUC distribution" in content
    assert "independently" in content
    assert "Limitations" in content


def test_null_experiment_rejects_too_few_observations(tmp_path):
    with pytest.raises(ValueError, match="at least 100"):
        run_null_experiments(
            n_observations=99,
            repetitions=100,
            seed=7,
            out_dir=tmp_path,
        )


def test_null_experiment_rejects_too_few_repetitions(tmp_path):
    with pytest.raises(ValueError, match="at least 20"):
        run_null_experiments(
            n_observations=300,
            repetitions=19,
            seed=7,
            out_dir=tmp_path,
        )
