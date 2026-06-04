"""Tests for the reporting math in ``benchmarks.core.reporting``.

Covers the aggregation, comparison-ratio, and artifact-writing logic that turns
fit results into report numbers, where a bug would silently corrupt output. The
fixtures are plain ``FitResult`` objects, so these tests do not build or run any
model.
"""

import json

import numpy as np
import pytest

from benchmarks.core.comparison import DEFAULT_METRICS, ComparisonSpec
from benchmarks.core.reporting import (
    aggregate_candidates,
    aggregate_parameter_estimates,
    aggregate_parameter_summaries,
    build_comparison,
    write_results,
)
from benchmarks.core.runner import FitMetrics, FitResult, McmcSettings, ParameterSummary

_SPEC = ComparisonSpec(
    name="demo",
    arms=("innovation", "state"),
    baseline="innovation",
    match_keys=("dataset", "innovation_sd", "autoreg"),
    metrics=DEFAULT_METRICS,
)


def _fit_result(
    candidate,
    arm,
    *,
    repeat=0,
    wall_time_s=10.0,
    ess_median=20.0,
    ess_min=5.0,
    divergences=0,
):
    """Create a small benchmark fit result for reporting tests.

    Returns
    -------
    FitResult
        Synthetic fit result with configurable metrics.
    """
    return FitResult(
        candidate=candidate,
        arm=arm,
        repeat=repeat,
        dataset="synthetic",
        config_fields={
            "parameterization": arm,
            "innovation_sd": 0.01,
            "autoreg": 0.9,
        },
        settings=McmcSettings(num_warmup=5, num_samples=7, num_chains=1, seed=42),
        metrics=FitMetrics(
            wall_time_s=wall_time_s,
            ess_per_sec_rt_median=ess_median,
            ess_per_sec_rt_min=ess_min,
            divergences=divergences,
            tree_depth_mean=3.0,
            tree_depth_max=4,
            ebfmi_min=0.5,
            rhat_rt_max=1.01,
        ),
        n_initialization_points=7,
        parameter_summaries=[
            ParameterSummary(
                site="example_site", index="", mean=1.5, ess=25.0, rhat=1.01
            )
        ],
    )


def test_aggregate_candidates_averages_repeats_and_pairs_state_with_innovation():
    """Candidate aggregation averages repeats and pairs comparable arms."""
    results = [
        _fit_result(
            "he_innovation", "innovation", repeat=0, wall_time_s=10.0, divergences=1
        ),
        _fit_result(
            "he_innovation",
            "innovation",
            repeat=1,
            wall_time_s=14.0,
            ess_median=30.0,
            ess_min=7.0,
            divergences=2,
        ),
        _fit_result(
            "he_state", "state", wall_time_s=6.0, ess_median=50.0, ess_min=12.0
        ),
    ]

    candidates = aggregate_candidates(results)
    comparison = build_comparison(candidates, _SPEC)

    innovation = next(r for r in candidates if r["candidate"] == "he_innovation")
    assert innovation["n_runs"] == 2
    assert innovation["wall_time_s"] == 12.0
    assert innovation["ess_per_sec_rt_median"] == 25.0
    assert innovation["divergences"] == 3

    assert len(comparison) == 1
    pair = comparison[0]
    assert pair["wall_time_s__ratio__state"] == 2.0
    assert pair["ess_per_sec_rt_median__ratio__state"] == 2.0
    assert pair["divergences__innovation"] == 3
    assert pair["divergences__state"] == 0


def test_aggregate_candidates_preserves_worst_case_diagnostics_across_repeats():
    """Worst-case diagnostics use min/max aggregation instead of means."""
    first = _fit_result("he_innovation", "innovation", repeat=0)
    second = _fit_result("he_innovation", "innovation", repeat=1)
    first.metrics.ebfmi_min = 0.8
    second.metrics.ebfmi_min = 0.2
    first.metrics.rhat_rt_max = 1.01
    second.metrics.rhat_rt_max = 1.2

    row = aggregate_candidates([first, second])[0]
    assert row["ebfmi_min"] == 0.2
    assert row["rhat_rt_max"] == 1.2


def test_build_comparison_skips_unmatched_groups():
    """Comparison rows are omitted when only the baseline arm is present."""
    candidates = aggregate_candidates([_fit_result("he_innovation", "innovation")])
    assert build_comparison(candidates, _SPEC) == []


def test_build_comparison_supports_single_arm_spec():
    """A single-arm spec produces no comparison rows but is accepted."""
    spec = ComparisonSpec(
        name="single",
        arms=("innovation",),
        baseline="innovation",
        match_keys=("dataset", "innovation_sd", "autoreg"),
        metrics=DEFAULT_METRICS,
    )
    candidates = aggregate_candidates([_fit_result("he_innovation", "innovation")])
    assert build_comparison(candidates, spec) == []


def test_aggregate_parameter_summaries_groups_sites_across_repeats():
    """Parameter site summaries aggregate ESS and R-hat across scalar elements."""
    first = _fit_result("he_innovation", "innovation", wall_time_s=10.0)
    first.parameter_summaries = [
        ParameterSummary("site_a", "[0]", mean=1.0, ess=20.0, rhat=1.01),
        ParameterSummary("site_a", "[1]", mean=2.0, ess=40.0, rhat=1.03),
        ParameterSummary("site_b", "", mean=3.0, ess=float("nan"), rhat=float("nan")),
    ]
    second = _fit_result("he_innovation", "innovation", repeat=1, wall_time_s=5.0)
    second.parameter_summaries = [
        ParameterSummary("site_a", "[0]", mean=1.5, ess=10.0, rhat=1.02),
    ]

    rows = aggregate_parameter_summaries([first, second])

    site_a = next(row for row in rows if row["site"] == "site_a")
    assert site_a["n_elements"] == 3
    assert site_a["ess_median"] == 20.0
    assert site_a["ess_min"] == 10.0
    assert site_a["ess_per_sec_median"] == 2.0
    assert site_a["rhat_max"] == 1.03

    site_b = next(row for row in rows if row["site"] == "site_b")
    assert site_b["n_finite_ess"] == 0
    assert np.isnan(site_b["ess_median"])


def test_aggregate_parameter_estimates_averages_mean_and_std_across_repeats():
    """Per-element posterior mean and std are averaged across repeats."""
    first = _fit_result("he_innovation", "innovation")
    first.parameter_summaries = [
        ParameterSummary("site_a", "[0]", mean=1.0, ess=20.0, rhat=1.01, std=0.4),
        ParameterSummary("site_a", "[1]", mean=2.0, ess=40.0, rhat=1.03, std=0.6),
    ]
    second = _fit_result("he_innovation", "innovation", repeat=1)
    second.parameter_summaries = [
        ParameterSummary("site_a", "[0]", mean=3.0, ess=10.0, rhat=1.02, std=0.8),
        ParameterSummary("site_a", "[1]", mean=2.0, ess=15.0, rhat=1.02, std=0.6),
    ]

    rows = aggregate_parameter_estimates([first, second])

    assert [row["index"] for row in rows] == ["[0]", "[1]"]
    element_0 = next(row for row in rows if row["index"] == "[0]")
    assert element_0["mean"] == pytest.approx(2.0)
    assert element_0["std"] == pytest.approx(0.6)


def test_write_results_creates_expected_artifacts(tmp_path):
    """Writing results creates CSV, JSON, and Markdown artifacts."""
    results = [
        _fit_result("he_innovation", "innovation"),
        _fit_result("he_state", "state", wall_time_s=5.0, ess_median=40.0),
    ]

    write_results(
        tmp_path,
        comparison_name="demo",
        results=results,
        spec=_SPEC,
        extra_payload={"prior_configs": {"demo": {"source": "def demo(): ..."}}},
    )

    expected = {
        "demo_runs.csv",
        "demo_candidates.csv",
        "demo_comparison.csv",
        "demo_parameters.csv",
        "demo_runs.json",
        "demo_report.md",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected

    payload = json.loads((tmp_path / "demo_runs.json").read_text())
    assert payload["suite"] == "demo"
    assert payload["arms"] == ["innovation", "state"]
    assert len(payload["comparison"]) == 1
    assert len(payload["parameters"]) == 2
    assert payload["prior_configs"]["demo"]["source"] == "def demo(): ..."

    parameter_rows = (tmp_path / "demo_parameters.csv").read_text()
    assert "site,index,mean,std,q025,q25,q50,q75,q975,ess,rhat" in parameter_rows

    report = (tmp_path / "demo_report.md").read_text()
    assert "## Parameter Estimates" in report
    assert "| candidate | dataset | arm | site | index | mean | std |" in report
    assert "## Comparison" in report
