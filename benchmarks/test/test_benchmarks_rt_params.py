"""Tests for the ``rt_params`` benchmark suite."""

import json
import sys
import types
from dataclasses import replace
from datetime import date

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from benchmarks.core.comparison import DEFAULT_METRICS, ComparisonSpec
from benchmarks.core.datasets import SYNTHETIC_HE_WEEKLY_HOSPITAL, SyntheticProvider
from benchmarks.core.priors import real_he_ed_day_of_week_prior, real_he_i0_prior
from benchmarks.core.real_data import (
    RealDataSpec,
    _build_bundle,
    _build_ed_visits_signal,
    _build_hospital_signal,
)
from benchmarks.core.reference_data import name_for_location, population_for_location
from benchmarks.core.reporting import (
    aggregate_candidates,
    aggregate_parameter_estimates,
    aggregate_parameter_summaries,
    build_comparison,
    format_bundle_summary,
    print_comparison_tables,
    print_data_summary,
    write_results,
)
from benchmarks.core.runner import FitMetrics, FitResult, McmcSettings, ParameterSummary
from benchmarks.core.signals import DatasetBundle, SignalSeries
from benchmarks.suites import rt_params
from benchmarks.suites.rt_params import BuildConfig, build_he_model


def _fit_result(
    candidate,
    parameterization,
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
        arm=parameterization,
        repeat=repeat,
        dataset="synthetic",
        config_fields={
            "parameterization": parameterization,
            "innovation_sd": 0.01,
            "autoreg": 0.9,
        },
        config=BuildConfig(
            parameterization=parameterization,
            innovation_sd=0.01,
            autoreg=0.9,
        ),
        settings=McmcSettings(
            num_warmup=5,
            num_samples=7,
            num_chains=1,
            seed=42,
        ),
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
                site="example_site",
                index="",
                mean=1.5,
                ess=25.0,
                rhat=1.01,
            )
        ],
    )


def test_parameterizations_are_centered_and_noncentered():
    """The suite compares exactly the innovation and state parameterizations."""
    assert rt_params.PARAMETERIZATIONS == ("innovation", "state")


def test_resolve_priors_handles_named_and_explicit_pairs():
    """Named and explicit prior arguments resolve to prior pairs."""
    assert rt_params._resolve_priors([]) == [rt_params.TIGHT_PRIOR]
    assert rt_params._resolve_priors(["tight"]) == [rt_params.TIGHT_PRIOR]
    assert rt_params._resolve_priors(["loose"]) == [rt_params.LOOSE_PRIOR]
    assert rt_params._resolve_priors(["both"]) == [
        rt_params.TIGHT_PRIOR,
        rt_params.LOOSE_PRIOR,
    ]
    assert rt_params._resolve_priors(["0.05,0.7"]) == [(0.05, 0.7)]


def test_resolve_priors_rejects_malformed_pair():
    """Malformed explicit prior pairs are rejected."""
    with pytest.raises(ValueError, match="Prior pair must be"):
        rt_params._resolve_priors(["0.05"])


@pytest.mark.parametrize(
    "prior,match",
    [
        ("0,0.7", "innovation sd must be positive"),
        ("0.05,1", "autoreg must satisfy"),
        ("0.05,-1", "autoreg must satisfy"),
    ],
)
def test_resolve_priors_rejects_invalid_domains(prior, match):
    """Explicit prior pairs must stay inside the supported parameter domain."""
    with pytest.raises(ValueError, match=match):
        rt_params._resolve_priors([prior])


def test_no_x64_argument_is_not_supported(monkeypatch):
    """The removed ``--no-x64`` CLI option is not accepted."""
    monkeypatch.setattr(sys, "argv", ["rt_params.py", "--no-x64"])
    with pytest.raises(SystemExit) as exc_info:
        rt_params._parse_args()
    assert exc_info.value.code == 2


def test_real_data_cli_requires_as_of(monkeypatch):
    """Real-data CLI runs require an ``--as-of`` date."""
    monkeypatch.setattr(sys, "argv", ["rt_params.py", "--data-source", "real"])
    with pytest.raises(SystemExit) as exc_info:
        rt_params._parse_args()
    assert exc_info.value.code == 2


def test_real_data_cli_parses_options(monkeypatch):
    """Real-data CLI options parse into the expected namespace values."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rt_params.py",
            "--data-source",
            "real",
            "--disease",
            "RSV",
            "--location",
            "CA",
            "--as-of",
            "2025-01-15",
            "--training-days",
            "120",
            "--omit-last-days",
            "3",
        ],
    )

    args = rt_params._parse_args()

    assert args.data_source == "real"
    assert args.disease == "RSV"
    assert args.location == "CA"
    assert args.as_of == date(2025, 1, 15)
    assert args.training_days == 120
    assert args.omit_last_days == 3


def _example_bundle():
    """Build a small single-signal bundle for summary tests.

    Returns
    -------
    DatasetBundle
        Bundle with one daily ED-visit signal containing a missing value.
    """
    return DatasetBundle(
        name="example",
        population_size=1234.0,
        obs_start_date=date(2025, 1, 1),
        n_days_post_init=2,
        signals={
            "ed_visits": SignalSeries(
                name="ed_visits",
                values=jnp.array([1.0, jnp.nan, 3.0]),
                cadence="daily",
                start_date=date(2025, 1, 1),
                extras={"delay_pmf": jnp.array([1.0])},
            )
        },
        gen_int_pmf=jnp.array([1.0]),
        fixed_params={"right_truncation_offset": 2},
    )


def test_signal_series_end_date_steps_by_cadence():
    """Regular signals derive end_date by stepping start_date by the cadence."""
    daily = SignalSeries(
        name="ed_visits",
        values=jnp.array([1.0, 2.0, 3.0]),
        cadence="daily",
        start_date=date(2025, 1, 1),
    )
    weekly = SignalSeries(
        name="hospital",
        values=jnp.array([1.0, 2.0, 3.0]),
        cadence="weekly",
        start_date=date(2025, 1, 1),
    )
    assert daily.end_date == date(2025, 1, 3)
    assert weekly.end_date == date(2025, 1, 15)


def test_signal_series_end_date_uses_times_for_irregular_signals():
    """Irregular signals offset start_date by the maximum times index."""
    signal = SignalSeries(
        name="wastewater",
        values=jnp.array([1.0, 2.0]),
        cadence="daily",
        start_date=date(2025, 1, 1),
        times=jnp.array([0, 9]),
    )
    assert signal.end_date == date(2025, 1, 10)


def test_format_bundle_summary_reports_shape_dates_and_missing():
    """Bundle summaries include signal shape, dates, and missing counts."""
    summary = format_bundle_summary(_example_bundle())

    assert "Dataset: example" in summary
    assert "signal: ed_visits" in summary
    assert "missing_or_nan: 1" in summary
    assert "date_range: 2025-01-01 to 2025-01-03" in summary


def test_print_data_summary(capsys):
    """Printed data summaries cover each bundle in the iterable."""
    print_data_summary([_example_bundle()])

    output = capsys.readouterr().out
    assert "Dataset: example" in output
    assert "signal: ed_visits" in output


def test_main_dry_run_data_exits_before_fitting(monkeypatch, capsys):
    """Dry-run data mode summarizes inputs and skips fitting."""
    bundle = DatasetBundle(
        name="example",
        population_size=1234.0,
        obs_start_date=date(2025, 1, 1),
        n_days_post_init=1,
        signals={},
        gen_int_pmf=jnp.array([1.0]),
    )

    monkeypatch.setattr(sys, "argv", ["rt_params.py", "--dry-run-data"])
    monkeypatch.setattr(rt_params, "load_he_bundle", lambda args: bundle)

    def fail_if_called(*args, **kwargs):
        """Fail the test if fitting is attempted."""
        raise AssertionError("run_comparison should not run for --dry-run-data")

    monkeypatch.setattr(rt_params, "run_comparison", fail_if_called)

    rt_params.main()

    assert "Dataset: example" in capsys.readouterr().out


def test_main_reports_real_data_loader_errors_without_traceback(monkeypatch):
    """Loader validation errors are surfaced as concise CLI failures."""
    monkeypatch.setattr(sys, "argv", ["rt_params.py", "--dry-run-data"])

    def fail_load(args):
        """Raise a loader-side validation error."""
        raise ValueError("bad real-data window")

    monkeypatch.setattr(rt_params, "load_he_bundle", fail_load)

    with pytest.raises(SystemExit) as exc_info:
        rt_params.main()

    assert str(exc_info.value) == "error: bad real-data window"


def test_main_reports_invalid_prior_without_traceback(monkeypatch):
    """Prior validation errors are surfaced as concise CLI failures."""
    monkeypatch.setattr(sys, "argv", ["rt_params.py", "--prior", "0,0.7"])

    with pytest.raises(SystemExit) as exc_info:
        rt_params.main()

    assert str(exc_info.value) == "error: Prior innovation sd must be positive; got 0"


def test_real_he_prior_helpers_are_benchmark_local():
    """Real H+E prior helpers return benchmark-local random variables."""
    i0_prior = real_he_i0_prior()
    dow_prior = real_he_ed_day_of_week_prior()

    assert i0_prior.name == "I0"
    assert dow_prior.name == "ed_day_of_week_effect"
    assert dow_prior.base_rv.name == "ed_day_of_week_effect_raw"


def test_build_he_model_wires_right_truncation_from_bundle():
    """H+E builder wires right-truncation PMFs from dataset metadata."""
    bundle = SyntheticProvider().get(SYNTHETIC_HE_WEEKLY_HOSPITAL)
    bundle = replace(
        bundle,
        fixed_params={
            **bundle.fixed_params,
            "right_truncation_pmf": jnp.array([0.25, 0.75]),
            "right_truncation_offset": 1,
        },
    )

    built = build_he_model(BuildConfig(parameterization="innovation"), bundle)

    assert built.model.observations["ed_visits"].right_truncation_rv is not None
    assert built.run_kwargs["ed_visits"]["right_truncation_offset"] == 1


def test_static_reference_data_covers_real_data_locations():
    """Static references provide benchmark-local location names and populations."""
    assert population_for_location("US") == 341784857
    assert population_for_location("CA") == 39355309
    assert name_for_location("CA") == "California"


def test_static_reference_data_rejects_unknown_values():
    """Unknown static reference keys fail with useful errors."""
    with pytest.raises(ValueError, match="No static population"):
        population_for_location("XX")


def test_aggregate_candidates_averages_repeats_and_pairs_state_with_innovation():
    """Candidate aggregation averages repeats and pairs comparable arms."""
    results = [
        _fit_result(
            "he_weekly_innovation",
            "innovation",
            repeat=0,
            wall_time_s=10.0,
            ess_median=20.0,
            ess_min=5.0,
            divergences=1,
        ),
        _fit_result(
            "he_weekly_innovation",
            "innovation",
            repeat=1,
            wall_time_s=14.0,
            ess_median=30.0,
            ess_min=7.0,
            divergences=2,
        ),
        _fit_result(
            "he_weekly_state",
            "state",
            wall_time_s=6.0,
            ess_median=50.0,
            ess_min=12.0,
            divergences=0,
        ),
    ]

    candidates = aggregate_candidates(results)
    comparison = build_comparison(candidates, rt_params.COMPARISON_SPEC)

    innovation = next(
        row for row in candidates if row["candidate"] == "he_weekly_innovation"
    )
    assert innovation["n_runs"] == 2
    assert innovation["wall_time_s"] == 12.0
    assert innovation["ess_per_sec_rt_median"] == 25.0
    assert innovation["ess_per_sec_rt_min"] == 6.0
    assert innovation["divergences"] == 3

    assert len(comparison) == 1
    pair = comparison[0]
    assert pair["wall_time_s__ratio__state"] == 2.0
    assert pair["ess_per_sec_rt_median__ratio__state"] == 2.0
    assert pair["ess_per_sec_rt_min__ratio__state"] == 2.0
    assert pair["divergences__innovation"] == 3
    assert pair["divergences__state"] == 0


def test_aggregate_candidates_preserves_worst_case_diagnostics_across_repeats():
    """Worst-case diagnostics use min/max aggregation instead of means."""
    first = _fit_result("he_weekly_innovation", "innovation", repeat=0)
    second = _fit_result("he_weekly_innovation", "innovation", repeat=1)
    first.metrics.ebfmi_min = 0.8
    second.metrics.ebfmi_min = 0.2
    first.metrics.rhat_rt_max = 1.01
    second.metrics.rhat_rt_max = 1.2

    candidates = aggregate_candidates([first, second])

    row = candidates[0]
    assert row["ebfmi_min"] == 0.2
    assert row["rhat_rt_max"] == 1.2


def test_build_comparison_skips_unmatched_groups():
    """Comparison rows are omitted when only the baseline arm is present."""
    candidates = aggregate_candidates(
        [_fit_result("he_weekly_innovation", "innovation")]
    )
    assert build_comparison(candidates, rt_params.COMPARISON_SPEC) == []


def test_build_comparison_supports_single_arm_spec():
    """A single-arm spec produces no comparison rows but is accepted."""
    spec = ComparisonSpec(
        name="single",
        arms=("innovation",),
        baseline="innovation",
        match_keys=("dataset", "innovation_sd", "autoreg"),
        metrics=DEFAULT_METRICS,
    )
    candidates = aggregate_candidates(
        [_fit_result("he_weekly_innovation", "innovation")]
    )
    assert build_comparison(candidates, spec) == []


def test_aggregate_parameter_summaries_groups_sites_across_repeats():
    """Parameter site summaries aggregate ESS and R-hat across scalar elements."""
    first = _fit_result("he_weekly_innovation", "innovation", wall_time_s=10.0)
    first.parameter_summaries = [
        ParameterSummary("site_a", "[0]", mean=1.0, ess=20.0, rhat=1.01),
        ParameterSummary("site_a", "[1]", mean=2.0, ess=40.0, rhat=1.03),
        ParameterSummary("site_b", "", mean=3.0, ess=float("nan"), rhat=float("nan")),
    ]
    second = _fit_result(
        "he_weekly_innovation",
        "innovation",
        repeat=1,
        wall_time_s=5.0,
    )
    second.parameter_summaries = [
        ParameterSummary("site_a", "[0]", mean=1.5, ess=10.0, rhat=1.02),
    ]

    rows = aggregate_parameter_summaries([first, second])

    site_a = next(row for row in rows if row["site"] == "site_a")
    assert site_a["candidate"] == "he_weekly_innovation"
    assert site_a["arm"] == "innovation"
    assert site_a["n_elements"] == 3
    assert site_a["n_finite_ess"] == 3
    assert site_a["ess_median"] == 20.0
    assert site_a["ess_min"] == 10.0
    assert site_a["ess_per_sec_median"] == 2.0
    assert site_a["ess_per_sec_min"] == 2.0
    assert site_a["rhat_max"] == 1.03

    site_b = next(row for row in rows if row["site"] == "site_b")
    assert site_b["n_elements"] == 1
    assert site_b["n_finite_ess"] == 0
    assert np.isnan(site_b["ess_median"])
    assert np.isnan(site_b["rhat_max"])


def test_aggregate_parameter_estimates_averages_mean_and_std_across_repeats():
    """Per-element posterior mean and std are averaged across repeats."""
    first = _fit_result("he_weekly_innovation", "innovation")
    first.parameter_summaries = [
        ParameterSummary("site_a", "[0]", mean=1.0, ess=20.0, rhat=1.01, std=0.4),
        ParameterSummary("site_a", "[1]", mean=2.0, ess=40.0, rhat=1.03, std=0.6),
    ]
    second = _fit_result("he_weekly_innovation", "innovation", repeat=1)
    second.parameter_summaries = [
        ParameterSummary("site_a", "[0]", mean=3.0, ess=10.0, rhat=1.02, std=0.8),
        ParameterSummary("site_a", "[1]", mean=2.0, ess=15.0, rhat=1.02, std=0.6),
    ]

    rows = aggregate_parameter_estimates([first, second])

    assert [row["index"] for row in rows] == ["[0]", "[1]"]
    element_0 = next(row for row in rows if row["index"] == "[0]")
    assert element_0["mean"] == pytest.approx(2.0)
    assert element_0["std"] == pytest.approx(0.6)
    element_1 = next(row for row in rows if row["index"] == "[1]")
    assert element_1["mean"] == pytest.approx(2.0)
    assert element_1["std"] == pytest.approx(0.6)


def test_print_comparison_tables_includes_parameter_site_summary(capsys):
    """Console benchmark summaries include per-site parameter ESS."""
    results = [
        _fit_result("he_weekly_innovation", "innovation"),
        _fit_result("he_weekly_state", "state", wall_time_s=5.0, ess_median=40.0),
    ]
    results[0].parameter_summaries = [
        ParameterSummary("example_site", "", mean=1.5, ess=12345.0, rhat=1.01),
    ]

    print_comparison_tables(results, rt_params.COMPARISON_SPEC)

    output = capsys.readouterr().out
    assert "state benefit" in output
    assert "--- Parameter ESS by site ---" in output
    assert "example_site" in output
    assert "12345" in output
    assert "e+" not in output
    assert "ESS/s med" in output
    assert "finite" not in output
    assert output.count("-" * 116) == 2


def test_print_comparison_tables_includes_parameters_without_comparison(capsys):
    """Suites with no comparable group still print parameter-site summaries."""
    print_comparison_tables(
        [_fit_result("he_weekly_innovation", "innovation")],
        rt_params.COMPARISON_SPEC,
    )

    output = capsys.readouterr().out
    assert "No comparable arm groups to summarize." in output
    assert "--- Parameter ESS by site ---" in output
    assert "example_site" in output


def test_write_results_creates_expected_artifacts(tmp_path):
    """Writing results creates CSV, JSON, and Markdown artifacts."""
    results = [
        _fit_result("he_weekly_innovation", "innovation"),
        _fit_result("he_weekly_state", "state", wall_time_s=5.0, ess_median=40.0),
    ]

    write_results(
        tmp_path,
        comparison_name="rt_params",
        results=results,
        spec=rt_params.COMPARISON_SPEC,
        extra_payload={"prior_configs": {"demo": {"source": "def demo(): ..."}}},
    )

    expected = {
        "rt_params_runs.csv",
        "rt_params_candidates.csv",
        "rt_params_comparison.csv",
        "rt_params_parameters.csv",
        "rt_params_runs.json",
        "rt_params_report.md",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected

    payload = json.loads((tmp_path / "rt_params_runs.json").read_text())
    assert payload["suite"] == "rt_params"
    assert payload["arms"] == ["innovation", "state"]
    assert payload["baseline"] == "innovation"
    assert len(payload["runs"]) == 2
    assert len(payload["candidates"]) == 2
    assert len(payload["comparison"]) == 1
    assert len(payload["parameters"]) == 2
    assert len(payload["parameter_sites"]) == 2
    assert payload["parameters"][0]["site"] == "example_site"
    assert payload["parameter_sites"][0]["site"] == "example_site"
    assert payload["prior_configs"]["demo"]["source"] == "def demo(): ..."

    parameter_rows = (tmp_path / "rt_params_parameters.csv").read_text()
    assert "site,index,mean,std,q025,q25,q50,q75,q975,ess,rhat" in parameter_rows

    report = (tmp_path / "rt_params_report.md").read_text()
    assert "# rt_params benchmark" in report
    assert "## Candidates" in report
    assert "## Parameter Estimates" in report
    assert "| candidate | dataset | arm | site | index | mean | std |" in report
    assert "## Comparison" in report
    assert "## Parameter ESS by Site" in report
    assert "ess_per_sec_median" in report


def test_real_data_ed_signal_uses_current_nssp_schema(monkeypatch):
    """ED signal builder reads the current NSSP column schema."""
    calls = {}

    def get_nssp(**kwargs):
        """Return a minimal NSSP frame in the current schema.

        Returns
        -------
        polars.DataFrame
            Minimal NSSP rows for RSV and total ED visits.
        """
        calls.update(kwargs)
        return pl.DataFrame(
            {
                "reference_date": [
                    date(2025, 1, 1),
                    date(2025, 1, 1),
                    date(2025, 1, 2),
                    date(2025, 1, 2),
                ],
                "disease": ["RSV", "Total", "RSV", "Total"],
                "geo_value": ["US", "US", "US", "US"],
                "value": [10.0, 100.0, 12.0, 110.0],
            }
        )

    monkeypatch.setitem(
        sys.modules, "cfa.stf.data", types.SimpleNamespace(get_nssp=get_nssp)
    )

    signal = _build_ed_visits_signal(
        disease="RSV",
        loc_abbr="US",
        as_of=date(2025, 1, 10),
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 2),
        delay_pmf=jnp.array([1.0]),
    )

    assert calls["disease"] == ["RSV", "Total"]
    assert calls["lazy"] is False
    assert signal.start_date == date(2025, 1, 1)
    np.testing.assert_array_equal(np.asarray(signal.values), np.array([10.0, 12.0]))
    np.testing.assert_array_equal(
        np.asarray(signal.extras["other_ed_visits"]),
        np.array([90.0, 98.0]),
    )


def test_real_data_hospital_signal_uses_current_nhsn_schema(monkeypatch):
    """Hospital signal builder reads the current NHSN column schema."""
    calls = {}

    def get_nhsn_hrd(**kwargs):
        """Return a minimal NHSN HRD frame in the current schema.

        Returns
        -------
        polars.DataFrame
            Minimal NHSN hospital admission rows.
        """
        calls.update(kwargs)
        return pl.DataFrame(
            {
                "weekendingdate": [date(2025, 1, 4), date(2025, 1, 11)],
                "jurisdiction": ["US", "US"],
                "disease": ["RSV", "RSV"],
                "hospital_admissions": [40.0, 45.0],
            }
        )

    monkeypatch.setitem(
        sys.modules,
        "cfa.stf.data",
        types.SimpleNamespace(get_nhsn_hrd=get_nhsn_hrd),
    )

    signal = _build_hospital_signal(
        disease="RSV",
        loc_abbr="US",
        as_of=date(2025, 1, 15),
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 14),
        delay_pmf=jnp.array([1.0]),
    )

    assert calls["lazy"] is False
    assert signal.start_date == date(2025, 1, 4)
    np.testing.assert_array_equal(np.asarray(signal.values), np.array([40.0, 45.0]))


def test_real_data_bundle_rejects_pre_nhsn_hospital_window():
    """Hospital bundles fail before feed calls when the window predates NHSN."""
    with pytest.raises(ValueError, match="as_of >= 2024-11-12"):
        _build_bundle(
            "real_he",
            RealDataSpec(
                disease="COVID-19",
                loc_abbr="US",
                as_of=date(2024, 11, 1),
                n_training_days=150,
                n_days_to_omit=2,
            ),
        )


def test_real_data_bundle_uses_static_references_and_live_he_feeds(monkeypatch):
    """Bundle setup uses local populations and live disease-specific PMFs."""
    calls = {"nssp": 0, "nhsn": 0, "gen_int": 0, "delay": 0}

    def get_nssp(**kwargs):  # numpydoc ignore=RT01
        """Return a minimal NSSP frame for bundle construction."""
        calls["nssp"] += 1
        return pl.DataFrame(
            {
                "reference_date": [
                    date(2025, 1, 1),
                    date(2025, 1, 1),
                    date(2025, 1, 2),
                    date(2025, 1, 2),
                ],
                "disease": ["RSV", "Total", "RSV", "Total"],
                "value": [10.0, 100.0, 12.0, 110.0],
            }
        )

    def get_nhsn_hrd(**kwargs):  # numpydoc ignore=RT01
        """Return a minimal NHSN frame for bundle construction."""
        calls["nhsn"] += 1
        return pl.DataFrame(
            {
                "weekendingdate": [date(2025, 1, 4)],
                "hospital_admissions": [40.0],
            }
        )

    def get_nnh_generation_interval_pmf(**kwargs):  # numpydoc ignore=RT01
        """Return a disease-specific generation interval test PMF."""
        calls["gen_int"] += 1
        assert kwargs["disease"] == "RSV"
        return [0.2, 0.8]

    def get_nnh_delay_pmf(**kwargs):  # numpydoc ignore=RT01
        """Return a disease-specific delay test PMF."""
        calls["delay"] += 1
        assert kwargs["disease"] == "RSV"
        return [0.1, 0.9]

    def fail_if_called(*args, **kwargs):
        """Fail if the old R location helper call reappears."""
        raise AssertionError("R forecasttools location helper should not be called")

    monkeypatch.setitem(
        sys.modules,
        "cfa.stf.data",
        types.SimpleNamespace(
            get_nssp=get_nssp,
            get_nhsn_hrd=get_nhsn_hrd,
            get_nnh_delay_pmf=get_nnh_delay_pmf,
            get_nnh_generation_interval_pmf=get_nnh_generation_interval_pmf,
            get_nnh_right_truncation_pmf=fail_if_called,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "cfa.stf.forecasttools",
        types.SimpleNamespace(get_us_loc_pop_tbl=fail_if_called),
    )

    bundle = _build_bundle(
        "real_he",
        RealDataSpec(
            disease="RSV",
            loc_abbr="CA",
            as_of=date(2025, 1, 10),
            n_training_days=2,
            n_days_to_omit=0,
        ),
    )

    assert calls == {"nssp": 1, "nhsn": 1, "gen_int": 1, "delay": 1}
    assert bundle.population_size == 39355309
    assert bundle.fixed_params == {}
    assert sorted(bundle.signals) == ["ed_visits", "hospital"]
    np.testing.assert_array_equal(np.asarray(bundle.gen_int_pmf), np.array([0.2, 0.8]))
    np.testing.assert_array_equal(
        np.asarray(bundle.signals["ed_visits"].extras["delay_pmf"]),
        np.array([0.1, 0.9]),
    )
