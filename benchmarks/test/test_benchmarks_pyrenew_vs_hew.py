"""Tests for the pyrenew_vs_hew benchmark suite.

These tests build models but never run MCMC, and the HEW candidate's build
callable is not invoked, so no external package is required.
"""

import sys
import types

import pytest

from benchmarks.core.datasets import SYNTHETIC_HE_WEEKLY_HOSPITAL, SyntheticProvider
from benchmarks.core.hew_model import HEW_RT_SITE_NAMES
from benchmarks.core.models import BuiltFit
from benchmarks.core.runner import Candidate, McmcSettings, fit_candidate
from benchmarks.suites import pyrenew_vs_hew


def test_comparison_spec_pairs_hew_baseline_with_pyrenew():
    """The spec compares the HEW baseline against the PyRenew arm."""
    spec = pyrenew_vs_hew.COMPARISON_SPEC
    assert spec.arms == ("hew", "pyrenew-state")
    assert spec.baseline == "hew"
    assert spec.match_keys == ("dataset",)


def test_build_pyrenew_state_omits_day_of_week():
    """The PyRenew candidate builds an ED observation with no day-of-week effect."""
    bundle = SyntheticProvider().get(SYNTHETIC_HE_WEEKLY_HOSPITAL)

    built = pyrenew_vs_hew._build_pyrenew_state(bundle)

    assert isinstance(built, BuiltFit)
    assert built.model.observations["ed_visits"].day_of_week_rv is None
    assert built.dataset_name == bundle.name
    assert set(built.run_kwargs) >= {
        "n_days_post_init",
        "population_size",
        "obs_start_date",
        "hospital",
        "ed_visits",
    }


def test_build_candidates_pairs_hew_and_pyrenew(tmp_path):
    """Candidate assembly yields the HEW and PyRenew arms and writes the HEW dir."""
    bundle = SyntheticProvider().get(SYNTHETIC_HE_WEEKLY_HOSPITAL)
    args = types.SimpleNamespace(
        data_source="synthetic",
        model_dir=tmp_path / "hew",
        pyrenew_multisignal_dir=tmp_path / "pms",
        cfa_stf_dir=tmp_path / "cfa",
    )

    candidates = pyrenew_vs_hew._build_candidates(args, bundle)

    assert [c.arm for c in candidates] == ["hew", "pyrenew-state"]
    hew, pyr = candidates
    assert hew.rt_site_names == HEW_RT_SITE_NAMES
    assert hew.config_fields["ed_day_of_week"] is True
    assert pyr.config_fields["ed_day_of_week"] is False
    assert pyr.config_fields["model_family"] == "pyrenew"
    assert (args.model_dir / "data" / "model_params.json").is_file()


def test_fit_candidate_builds_model_and_forwards_metadata(monkeypatch):
    """fit_candidate calls build() once and forwards the candidate metadata."""
    captured: dict = {}
    sentinel_built = object()

    def fake_fit_and_measure(**kwargs):
        """Capture forwarded arguments instead of fitting.

        Returns
        -------
        str
            Sentinel result.
        """
        captured.update(kwargs)
        return "fit-result"

    from benchmarks.core import runner

    monkeypatch.setattr(runner, "fit_and_measure", fake_fit_and_measure)

    build_calls = {"n": 0}

    def build():
        """Count build invocations and return a sentinel.

        Returns
        -------
        object
            Sentinel built model.
        """
        build_calls["n"] += 1
        return sentinel_built

    candidate = Candidate(
        name="cand",
        arm="hew",
        config_fields={"model_family": "hew"},
        build=build,
        rt_site_names=("rt",),
    )
    settings = McmcSettings(num_warmup=1, num_samples=1, num_chains=1, seed=0)

    result = fit_candidate(candidate, settings, repeat=2)

    assert result == "fit-result"
    assert build_calls["n"] == 1
    assert captured["built"] is sentinel_built
    assert captured["arm"] == "hew"
    assert captured["config_fields"] == {"model_family": "hew"}
    assert captured["rt_site_names"] == ("rt",)
    assert captured["repeat"] == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
