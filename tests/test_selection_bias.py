
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

    assert report["experiment"] == (
        "Random-candidate selection bias simulation"
    )
    assert len(report["repetitions"]) == 30

    assert (tmp_path / "latest.json").is_file()
    assert (tmp_path / "latest.md").is_file()


def test_selection_bias_shows_selection_uplift(tmp_path):
    report = run_selection_bias(
        n_observations=300,
        n_trials=30,
        repetitions=50,
        seed=19,
        out_dir=tmp_path,
    )

    summary = report["summary"]

    assert summary["mean_selected_candidate_auc"] > (
        summary["mean_single_candidate_auc"]
    )

    assert summary["mean_selection_uplift"] > 0.0


def test_selection_bias_is_reproducible(tmp_path):
    first = run_selection_bias(
        n_observations=300,
        n_trials=20,
        repetitions=30,
        seed=42,
        out_dir=tmp_path / "first",
    )

    second = run_selection_bias(
        n_observations=300,
        n_trials=20,
        repetitions=30,
        seed=42,
        out_dir=tmp_path / "second",
    )

    assert first == second


def test_selection_bias_json_is_valid(tmp_path):
    run_selection_bias(
        n_observations=300,
        n_trials=20,
        repetitions=30,
        seed=7,
        out_dir=tmp_path,
    )

    report = json.loads(
        (tmp_path / "latest.json").read_text(encoding="utf-8")
    )

    assert report["null_reference_auc"] if (
        "null_reference_auc" in report
    ) else report["summary"]["null_reference_auc"] == 0.5


def test_selection_bias_rejects_invalid_parameters(tmp_path):
    with pytest.raises(ValueError, match="at least 100"):
        run_selection_bias(
            n_observations=99,
            out_dir=tmp_path,
        )

    with pytest.raises(ValueError, match="at least 2"):
        run_selection_bias(
            n_trials=1,
            out_dir=tmp_path,
        )

    with pytest.raises(ValueError, match="at least 20"):
        run_selection_bias(
            repetitions=10,
            out_dir=tmp_path,
        )
