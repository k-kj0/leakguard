
import json

import pytest

from leakguard.selection_bias import run_selection_bias


def test_selection_bias_creates_reports(tmp_path):
    report = run_selection_bias(
        n_observations=300,
        n_trials=20,
        repetitions=30,
        seed=7,
        out_dir=tmp_path,
    )

    assert isinstance(report, dict)
    assert "summary" in report
    assert "repetitions" in report

    assert (tmp_path / "latest.json").is_file()
    assert (tmp_path / "latest.md").is_file()


def test_selection_bias_detects_positive_selection_uplift(tmp_path):
    report = run_selection_bias(
        n_observations=300,
        n_trials=30,
        repetitions=50,
        seed=19,
        out_dir=tmp_path,
    )

    summary = report["summary"]

    assert summary["null_reference_auc"] == pytest.approx(0.5)
    assert summary["mean_selected_auc"] > summary["null_reference_auc"]


def test_selection_bias_is_reproducible(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = run_selection_bias(
        n_observations=300,
        n_trials=20,
        repetitions=30,
        seed=42,
        out_dir=first_dir,
    )

    second = run_selection_bias(
        n_observations=300,
        n_trials=20,
        repetitions=30,
        seed=42,
        out_dir=second_dir,
    )

    assert first == second


def test_selection_bias_json_is_valid(tmp_path):
    report = run_selection_bias(
        n_observations=300,
        n_trials=20,
        repetitions=30,
        seed=7,
        out_dir=tmp_path,
    )

    saved = json.loads(
        (tmp_path / "latest.json").read_text(encoding="utf-8")
    )

    assert saved == report
    assert report["summary"]["null_reference_auc"] == pytest.approx(0.5)


def test_selection_bias_rejects_invalid_parameters(tmp_path):
    with pytest.raises(ValueError):
        run_selection_bias(
            n_observations=0,
            n_trials=20,
            repetitions=30,
            seed=7,
            out_dir=tmp_path,
        )

    with pytest.raises(ValueError):
        run_selection_bias(
            n_observations=300,
            n_trials=0,
            repetitions=30,
            seed=7,
            out_dir=tmp_path,
        )

    with pytest.raises(ValueError):
        run_selection_bias(
            n_observations=300,
            n_trials=20,
            repetitions=0,
            seed=7,
            out_dir=tmp_path,
        )

