
import json

import pytest

from leakguard.redteam import run_red_team


def test_red_team_v2_all_challenges_pass(tmp_path):
    report = run_red_team(
        n=1200,
        seed=7,
        out_dir=tmp_path,
    )

    summary = report["summary"]

    assert report["version"] == 2
    assert summary["all_passed"] is True
    assert summary["challenge_cases"] == 11
    assert summary["passed"] == 11
    assert summary["failed"] == 0
    assert summary["pass_rate"] == 1.0
    assert summary["clean_controls_passed"] is True


def test_red_team_v2_covers_expected_failure_modes(tmp_path):
    report = run_red_team(
        n=1200,
        seed=7,
        out_dir=tmp_path,
    )

    cases = {
        item["case"]: item
        for item in report["cases"]
    }

    expected = {
        "duplicate_timestamp",
        "out_of_order_timestamp",
        "missing_market_value",
        "crossed_order_book",
        "negative_order_size",
        "non_positive_price",
        "invalid_depth_order",
        "large_timestamp_gap",
        "infinite_market_value",
        "zero_order_size_warning",
        "future_information_leakage",
    }

    assert set(cases) == expected

    for case in cases.values():
        assert case["passed"] is True, case


def test_red_team_v2_keeps_clean_controls(tmp_path):
    report = run_red_team(
        n=1200,
        seed=7,
        out_dir=tmp_path,
    )

    controls = {
        item["case"]: item
        for item in report["controls"]
    }

    assert set(controls) == {
        "clean_market_data_control",
        "clean_feature_control",
    }

    assert all(item["passed"] for item in controls.values())


def test_red_team_v2_is_reproducible(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    run_red_team(
        n=1200,
        seed=42,
        out_dir=first_dir,
    )

    run_red_team(
        n=1200,
        seed=42,
        out_dir=second_dir,
    )

    first = json.loads(
        (first_dir / "latest.json").read_text(encoding="utf-8")
    )

    second = json.loads(
        (second_dir / "latest.json").read_text(encoding="utf-8")
    )

    assert first == second


def test_red_team_v2_writes_both_reports(tmp_path):
    run_red_team(
        n=1200,
        seed=7,
        out_dir=tmp_path,
    )

    json_report = tmp_path / "latest.json"
    markdown_report = tmp_path / "latest.md"

    assert json_report.is_file()
    assert markdown_report.is_file()

    content = markdown_report.read_text(encoding="utf-8")

    assert "Leakguard Red-Team v2 Report" in content
    assert "large_timestamp_gap" in content
    assert "future_information_leakage" in content
    assert "not a general estimate" in content


def test_red_team_rejects_too_few_rows(tmp_path):
    with pytest.raises(ValueError, match="at least 500"):
        run_red_team(
            n=499,
            seed=7,
            out_dir=tmp_path,
        )
