
from leakguard.redteam import run_red_team


def test_red_team_challenge_suite_passes(tmp_path):
    report = run_red_team(
        n=1200,
        seed=7,
        out_dir=tmp_path,
    )

    summary = report["summary"]

    assert summary["all_passed"] is True
    assert summary["challenge_cases"] == 6
    assert summary["passed"] == 6
    assert summary["failed"] == 0
    assert summary["pass_rate"] == 1.0
    assert summary["clean_controls_passed"] is True


def test_red_team_covers_expected_failure_modes(tmp_path):
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
        "future_information_leakage",
    }

    assert set(cases) == expected

    for case in cases.values():
        assert case["passed"] is True


def test_red_team_writes_reproducible_reports(tmp_path):
    run_red_team(
        n=1200,
        seed=7,
        out_dir=tmp_path,
    )

    assert (tmp_path / "latest.json").is_file()
    assert (tmp_path / "latest.md").is_file()
