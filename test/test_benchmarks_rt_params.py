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

from benchmarks.core.datasets import SYNTHETIC_HE_WEEKLY_HOSPITAL, SyntheticProvider
from benchmarks.core.models import BuildConfig, build_he_model
from benchmarks.core.priors import real_he_ed_day_of_week_prior, real_he_i0_prior
from benchmarks.core.real_data import _build_ed_visits_signal, _build_hospital_signal
from benchmarks.core.reporting import aggregate_results, write_results
from benchmarks.core.runner import FitMetrics, FitResult, McmcSettings
from benchmarks.core.signals import DatasetBundle, SignalSeries
from benchmarks.suites import rt_params


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
        repeat=repeat,
        dataset="synthetic",
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


def test_load_bundles_uses_real_data_provider_for_real_he(monkeypatch):
    """Real H+E candidates load through the real-data provider."""
    bundle = object()
    captured_specs = {}

    class FakeRealDataProvider:
        """Minimal provider that captures requested real-data specs."""

        def __init__(self, specs):
            """Store the provided specs in the outer capture mapping."""
            captured_specs.update(specs)

        def get(self, name):
            """Return the fake bundle for the expected real-data name.

            Returns
            -------
            object
                Fake bundle supplied by the test.
            """
            assert name == rt_params.REAL_HE_DATASET
            return bundle

    monkeypatch.setattr(rt_params, "RealDataProvider", FakeRealDataProvider)
    args = types.SimpleNamespace(
        data_source="real",
        disease="RSV",
        location="CA",
        as_of=date(2025, 1, 15),
        training_days=120,
        omit_last_days=3,
    )

    bundles = rt_params._load_bundles(args)

    assert bundles == {rt_params.SYNTHETIC_HE_WEEKLY_HOSPITAL: bundle}
    spec = captured_specs[rt_params.REAL_HE_DATASET]
    assert spec.disease == "RSV"
    assert spec.loc_abbr == "CA"
    assert spec.as_of == date(2025, 1, 15)
    assert spec.n_training_days == 120
    assert spec.n_days_to_omit == 3
    assert spec.signals == ("hospital", "ed_visits")


def test_print_data_summary(capsys):
    """Data summaries include signal shape, dates, and missing counts."""
    bundle = DatasetBundle(
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

    rt_params._print_data_summary({"example": bundle})

    output = capsys.readouterr().out
    assert "Dataset: example" in output
    assert "signal: ed_visits" in output
    assert "missing_or_nan: 1" in output
    assert "date_range: 2025-01-01 to 2025-01-03" in output


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
    monkeypatch.setattr(
        rt_params,
        "_load_bundles",
        lambda args: {rt_params.SYNTHETIC_HE_WEEKLY_HOSPITAL: bundle},
    )

    def fail_if_called(*args, **kwargs):
        """Fail the test if fitting is attempted."""
        raise AssertionError("fit_and_measure should not run for --dry-run-data")

    monkeypatch.setattr(rt_params, "fit_and_measure", fail_if_called)

    rt_params.main()

    assert "Dataset: example" in capsys.readouterr().out


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


def test_aggregate_results_averages_repeats_and_pairs_state_with_innovation():
    """Aggregate results average repeats and pair comparable candidates."""
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

    candidates, pairs = aggregate_results(results)

    innovation = next(
        row for row in candidates if row["candidate"] == "he_weekly_innovation"
    )
    assert innovation["n_runs"] == 2
    assert innovation["wall_time_s"] == 12.0
    assert innovation["ess_per_sec_rt_median"] == 25.0
    assert innovation["ess_per_sec_rt_min"] == 6.0
    assert innovation["divergences_total"] == 3

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["wall_s_ratio"] == 0.5
    assert pair["ess_per_s_med_ratio"] == 2.0
    assert pair["ess_per_s_min_ratio"] == 2.0
    assert pair["divergences_innov"] == 3
    assert pair["divergences_state"] == 0


def test_aggregate_results_skips_unmatched_pairs():
    """Aggregate results omit pair rows without both parameterizations."""
    _, pairs = aggregate_results([_fit_result("he_weekly_innovation", "innovation")])
    assert pairs == []


def test_write_results_creates_expected_artifacts(tmp_path):
    """Writing results creates CSV, JSON, and Markdown artifacts."""
    results = [
        _fit_result("he_weekly_innovation", "innovation"),
        _fit_result("he_weekly_state", "state", wall_time_s=5.0, ess_median=40.0),
    ]

    write_results(tmp_path, suite_name="rt_params", results=results)

    expected = {
        "rt_params_runs.csv",
        "rt_params_candidates.csv",
        "rt_params_pairs.csv",
        "rt_params_runs.json",
        "rt_params_report.md",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected

    payload = json.loads((tmp_path / "rt_params_runs.json").read_text())
    assert payload["suite"] == "rt_params"
    assert len(payload["runs"]) == 2
    assert len(payload["candidates"]) == 2
    assert len(payload["pairs"]) == 1

    report = (tmp_path / "rt_params_report.md").read_text()
    assert "# rt_params benchmark" in report
    assert "## Candidates" in report
    assert "## State vs Innovation" in report


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
