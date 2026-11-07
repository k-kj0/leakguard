
"""Controlled adversarial challenges for Leakguard research validation."""

import argparse
import json
from pathlib import Path

import numpy as np

from .checks import check_causality, check_data_integrity
from .data_quality import audit_dataframe
from .features import build_features, build_features_leaky
from .simulate import simulate_order_book


def _make_challenges(clean):
    """Return corrupted datasets with known expected audit findings."""
    cases = []

    def add(case_id, mutate, expected_code, severity="error",
            expected_interval_ms=None):
        corrupted = clean.copy(deep=True)
        mutate(corrupted)
        cases.append({
            "case": case_id,
            "data": corrupted,
            "expected_issue_code": expected_code,
            "expected_severity": severity,
            "expected_interval_ms": expected_interval_ms,
        })

    add(
        "duplicate_timestamp",
        lambda df: df.__setitem__(
            "ts_ms",
            df["ts_ms"].where(
                df.index != 101,
                df.loc[100, "ts_ms"],
            ),
        ),
        "duplicate_timestamps",
    )

    def reorder(df):
        first = df.loc[100, "ts_ms"]
        second = df.loc[101, "ts_ms"]
        df.loc[100, "ts_ms"] = second
        df.loc[101, "ts_ms"] = first

    add("out_of_order_timestamp", reorder, "out_of_order_timestamps")

    add(
        "missing_market_value",
        lambda df: df.loc.__setitem__((100, "bid_sz_1"), np.nan),
        "missing_or_non_numeric_values",
    )

    add(
        "crossed_order_book",
        lambda df: df.loc.__setitem__(
            (100, "bid_px_1"),
            df.loc[100, "ask_px_1"] + 0.01,
        ),
        "crossed_or_locked_books",
    )

    add(
        "negative_order_size",
        lambda df: df.loc.__setitem__((100, "bid_sz_1"), -1.0),
        "negative_sizes",
    )

    add(
        "non_positive_price",
        lambda df: df.loc.__setitem__((100, "bid_px_1"), 0.0),
        "non_positive_prices",
    )

    add(
        "invalid_depth_order",
        lambda df: df.loc.__setitem__(
            (100, "bid_px_2"),
            df.loc[100, "bid_px_1"] + 0.001,
        ),
        "invalid_depth_order",
    )

    def create_gap(df):
        df.loc[df.index[-1], "ts_ms"] += 10_000

    add(
        "large_timestamp_gap",
        create_gap,
        "timestamp_gaps",
        severity="warning",
        expected_interval_ms=50,
    )

    add(
        "infinite_market_value",
        lambda df: df.loc.__setitem__(
            (100, "ask_sz_1"),
            np.inf,
        ),
        "non_finite_values",
    )

    add(
        "zero_order_size_warning",
        lambda df: df.loc.__setitem__((100, "bid_sz_2"), 0.0),
        "zero_or_negative_sizes",
        severity="warning",
    )

    return cases


def _run_data_challenges(clean):
    """Run each corrupted dataset through the independent quality auditor."""
    results = []

    for challenge in _make_challenges(clean):
        audit = audit_dataframe(
            challenge["data"],
            expected_interval_ms=challenge["expected_interval_ms"],
        )

        matching = [
            issue
            for issue in audit["issues"]
            if issue["code"] == challenge["expected_issue_code"]
            and issue["severity"] == challenge["expected_severity"]
        ]

        detected = bool(matching)

        results.append({
            "case": challenge["case"],
            "expected_issue_code": challenge["expected_issue_code"],
            "expected_severity": challenge["expected_severity"],
            "detected": detected,
            "observed_status": audit["status"],
            "observed_issues": audit["issues"],
            "passed": detected,
        })

    return results


def _run_causality_challenges(clean, seed):
    """Check that causal features pass and future-dependent features fail."""
    clean_issues = check_causality(
        build_features,
        clean,
        n_probes=20,
        seed=seed,
    )

    leaky_issues = check_causality(
        build_features_leaky,
        clean,
        n_probes=20,
        seed=seed,
    )

    clean_passed = len(clean_issues) == 0
    leakage_detected = "LEAK_future_ret" in leaky_issues

    return [
        {
            "case": "clean_feature_control",
            "expected_detection": "no future-dependent features",
            "detected": clean_passed,
            "observed_issues": clean_issues,
            "passed": clean_passed,
        },
        {
            "case": "future_information_leakage",
            "expected_detection": "LEAK_future_ret",
            "detected": leakage_detected,
            "observed_issues": leaky_issues,
            "passed": leakage_detected,
        },
    ]


def _run_clean_data_control(clean):
    """Ensure the known-clean synthetic fixture passes the basic integrity check."""
    issues = check_data_integrity(clean)
    passed = len(issues) == 0

    return {
        "case": "clean_market_data_control",
        "expected_detection": "no integrity issues",
        "detected": passed,
        "observed_issues": issues,
        "passed": passed,
    }


def render_report(report):
    """Render a readable Markdown report."""
    summary = report["summary"]

    lines = [
        "# Leakguard Red-Team v2 Report",
        "",
        f"- Random seed: **{report['seed']}**",
        f"- Synthetic rows: **{report['rows']}**",
        f"- Challenge cases: **{summary['challenge_cases']}**",
        f"- Passed: **{summary['passed']}**",
        f"- Failed: **{summary['failed']}**",
        f"- Challenge pass rate: **{summary['pass_rate']:.1%}**",
        f"- Clean controls passed: **{summary['clean_controls_passed']}**",
        (
            "- Overall result: **"
            + ("PASS" if summary["all_passed"] else "FAIL")
            + "**"
        ),
        "",
        "## Adversarial challenge results",
        "",
        "| Challenge | Expected finding | Detected | Passed |",
        "|---|---|---:|---:|",
    ]

    for case in report["cases"]:
        expected = case.get(
            "expected_issue_code",
            case.get("expected_detection", ""),
        )
        lines.append(
            f"| {case['case']} | {expected} | "
            f"{case['detected']} | {case['passed']} |"
        )

    lines += [
        "",
        "## Clean controls",
        "",
        "These controls check that known-clean inputs are not incorrectly rejected.",
        "",
    ]

    for control in report["controls"]:
        lines.append(
            f"- **{control['case']}:** "
            f"{'PASS' if control['passed'] else 'FAIL'}"
        )

    lines += [
        "",
        "## Interpretation and limitations",
        "",
        "- This is a controlled synthetic fault-injection suite.",
        "- The pass rate applies only to the listed, known challenge cases.",
        "- It is not a general estimate of detection recall or precision.",
        "- Clean controls cover the specific synthetic fixture and feature checks.",
        "- Timestamp-gap warnings depend on the configured interval threshold.",
        "- Passing does not prove that a live exchange feed is complete or error-free.",
        "- Passing does not establish predictive performance or profitability.",
        "- New failure modes require additional labelled challenge cases.",
        "",
    ]

    return "\n".join(lines)


def run_red_team(n=1200, seed=7, out_dir="results/redteam"):
    """Run deterministic data-integrity and feature-causality challenges."""
    if n < 500:
        raise ValueError(
            "Use at least 500 synthetic rows for this challenge suite."
        )

    clean = simulate_order_book(n=n, seed=seed)

    controls = [
        _run_clean_data_control(clean),
    ]

    causality_cases = _run_causality_challenges(clean, seed)
    controls.append(causality_cases[0])

    cases = (
        _run_data_challenges(clean)
        + [causality_cases[1]]
    )

    passed = sum(bool(case["passed"]) for case in cases)
    failed = len(cases) - passed
    controls_passed = all(
        control["passed"] for control in controls
    )

    summary = {
        "challenge_cases": len(cases),
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / len(cases),
        "clean_controls_passed": controls_passed,
        "all_passed": failed == 0 and controls_passed,
    }

    report = {
        "suite": "Leakguard Red-Team v2",
        "version": 2,
        "seed": seed,
        "rows": len(clean),
        "cases": cases,
        "controls": controls,
        "summary": summary,
        "interpretation": (
            "Synthetic challenge results only; not a general detection "
            "recall estimate or evidence of trading profitability."
        ),
    }

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "latest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    markdown = render_report(report)
    (output / "latest.md").write_text(
        markdown,
        encoding="utf-8",
    )

    print(markdown)

    if not summary["all_passed"]:
        raise RuntimeError(
            "Red-Team v2 failed. Inspect results/redteam/latest.json."
        )

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="results/redteam")

    args = parser.parse_args()

    run_red_team(
        n=args.n,
        seed=args.seed,
        out_dir=args.out,
    )
