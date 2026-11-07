
"""Controlled fault-injection challenges for Leakguard."""

import argparse
import json
from pathlib import Path

import numpy as np

from .checks import check_causality, check_data_integrity
from .features import build_features, build_features_leaky
from .simulate import simulate_order_book


def _make_integrity_challenges(clean):
    """Create corrupted copies with known expected detections."""

    challenges = []

    duplicate = clean.copy()
    duplicate.loc[101, "ts_ms"] = duplicate.loc[100, "ts_ms"]
    challenges.append((
        "duplicate_timestamp",
        duplicate,
        "duplicate timestamps",
    ))

    unordered = clean.copy()
    first = unordered.loc[100, "ts_ms"]
    second = unordered.loc[101, "ts_ms"]
    unordered.loc[100, "ts_ms"] = second
    unordered.loc[101, "ts_ms"] = first
    challenges.append((
        "out_of_order_timestamp",
        unordered,
        "timestamps are not sorted",
    ))

    missing = clean.copy()
    missing.loc[100, "bid_sz_1"] = np.nan
    challenges.append((
        "missing_market_value",
        missing,
        "NaN values present",
    ))

    crossed = clean.copy()
    crossed.loc[100, "bid_px_1"] = (
        crossed.loc[100, "ask_px_1"] + 0.01
    )
    challenges.append((
        "crossed_order_book",
        crossed,
        "crossed book",
    ))

    negative_size = clean.copy()
    negative_size.loc[100, "bid_sz_1"] = -1.0
    challenges.append((
        "negative_order_size",
        negative_size,
        "negative size",
    ))

    return challenges


def _run_integrity_challenges(clean):
    """Run corrupted datasets through the existing integrity checker."""

    results = []

    for case_id, corrupted, expected_issue in (
        _make_integrity_challenges(clean)
    ):
        observed = check_data_integrity(corrupted)

        detected = any(
            expected_issue in issue
            for issue in observed
        )

        results.append({
            "case": case_id,
            "expected_detection": expected_issue,
            "detected": detected,
            "observed_issues": observed,
            "passed": detected,
        })

    return results


def _run_causality_challenges(clean, seed):
    """Check clean features and deliberately future-leaking features."""

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

    leakage_detected = (
        "LEAK_future_ret" in leaky_issues
    )

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


def render_report(report):
    """Render a human-readable Markdown challenge report."""

    summary = report["summary"]

    lines = [
        "# Leakguard Red-Team Challenge Report",
        "",
        f"- Random seed: **{report['seed']}**",
        f"- Synthetic rows: **{report['rows']}**",
        f"- Challenges: **{summary['challenge_cases']}**",
        f"- Passed: **{summary['passed']}**",
        f"- Failed: **{summary['failed']}**",
        (
            f"- Challenge pass rate: "
            f"**{summary['pass_rate']:.1%}**"
        ),
        (
            "- Clean controls passed: **"
            + str(summary["clean_controls_passed"])
            + "**"
        ),
        (
            "- Overall result: **"
            + ("PASS" if summary["all_passed"] else "FAIL")
            + "**"
        ),
        "",
        "## Challenge results",
        "",
        "| Challenge | Detected | Passed |",
        "|---|---|---|",
    ]

    for case in report["cases"]:
        lines.append(
            f"| {case['case']} | "
            f"{case['detected']} | "
            f"{case['passed']} |"
        )

    lines += [
        "",
        "## Clean controls",
        "",
        "Clean controls check that valid inputs are not incorrectly flagged.",
        "",
    ]

    for control in report["controls"]:
        lines.append(
            f"- **{control['case']}:** "
            f"{'PASS' if control['passed'] else 'FAIL'}"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- This is a controlled synthetic challenge suite.",
        "- The challenge pass rate measures only these known test cases.",
        "- It is not a general estimate of detection recall.",
        "- Passing does not prove real market data is error-free.",
        "- Passing does not prove a trading model is profitable.",
        "- New fault types require new challenge cases and expected outcomes.",
        "",
    ]

    return "\n".join(lines)


def run_red_team(
    n=1200,
    seed=7,
    out_dir="results/redteam",
):
    """Run deterministic data-integrity and feature-causality challenges."""

    if n < 500:
        raise ValueError(
            "Use at least 500 synthetic rows for this challenge suite."
        )

    clean = simulate_order_book(n=n, seed=seed)

    # Clean data must not produce integrity warnings.
    clean_data_issues = check_data_integrity(clean)
    clean_data_passed = len(clean_data_issues) == 0

    integrity_cases = _run_integrity_challenges(clean)

    causality_cases = _run_causality_challenges(
        clean,
        seed,
    )

    cases = integrity_cases + [
        causality_cases[1]
    ]

    controls = [
        {
            "case": "clean_market_data",
            "observed_issues": clean_data_issues,
            "passed": clean_data_passed,
        },
        causality_cases[0],
    ]

    passed = sum(
        bool(case["passed"])
        for case in cases
    )

    failed = len(cases) - passed

    controls_passed = all(
        control["passed"]
        for control in controls
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
        "suite": "Leakguard Red-Team Challenge",
        "seed": seed,
        "rows": len(clean),
        "cases": cases,
        "controls": controls,
        "summary": summary,
    }

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "latest.json").write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
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
            "Red-Team Challenge failed. Inspect latest.json."
        )

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out",
        default="results/redteam",
    )

    args = parser.parse_args()

    run_red_team(
        n=args.n,
        seed=args.seed,
        out_dir=args.out,
    )
