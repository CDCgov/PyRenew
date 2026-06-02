"""Tests for the prior_regimes example driver.

These tests build models and inspect provenance but never run MCMC.
"""

import pytest

from benchmarks.core.datasets import SYNTHETIC_HE_WEEKLY_HOSPITAL, SyntheticProvider
from benchmarks.core.models import BuiltFit
from benchmarks.examples import run_prior_regimes


def test_comparison_spec_arms_are_the_regimes():
    """The spec compares the regimes against the example baseline."""
    spec = run_prior_regimes.COMPARISON_SPEC
    assert spec.arms == tuple(run_prior_regimes.REGIMES)
    assert spec.baseline == "example"
    assert spec.match_keys == ("dataset",)


def test_every_regime_supplies_required_slots():
    """Each regime provides exactly the prior slots the structure consumes."""
    for name, regime_fn in run_prior_regimes.REGIMES.items():
        assert set(regime_fn()) == set(run_prior_regimes.REQUIRED_SLOTS), name


def test_validate_regimes_passes():
    """Regime validation accepts the shipped regimes."""
    run_prior_regimes._validate_regimes()


def test_each_regime_builds_a_model():
    """Every regime assembles a valid BuiltFit with no ED day-of-week effect."""
    bundle = SyntheticProvider().get(SYNTHETIC_HE_WEEKLY_HOSPITAL)
    for name, regime_fn in run_prior_regimes.REGIMES.items():
        built = run_prior_regimes._build_he_model(bundle, regime_fn())
        assert isinstance(built, BuiltFit), name
        assert built.model.observations["ed_visits"].day_of_week_rv is None, name
        assert set(built.run_kwargs) >= {"hospital", "ed_visits", "population_size"}


def test_build_candidates_one_per_regime():
    """Candidate assembly yields one candidate per regime, labeled by name."""
    bundle = SyntheticProvider().get(SYNTHETIC_HE_WEEKLY_HOSPITAL)
    candidates = run_prior_regimes.build_candidates(bundle)
    assert [c.arm for c in candidates] == list(run_prior_regimes.REGIMES)
    for candidate in candidates:
        assert candidate.config_fields == {"prior_config": candidate.name}


def test_prior_provenance_captures_inline_source():
    """Provenance records each regime's full source, including inline priors."""
    provenance = run_prior_regimes.prior_provenance()
    assert set(provenance) == set(run_prior_regimes.REGIMES)
    example = provenance["example"]
    assert example["module"] == "benchmarks.examples.run_prior_regimes"
    assert "def example_priors" in example["source"]
    assert "Beta(1.0, 10.0)" in example["source"]


def test_validate_regimes_rejects_missing_slot(monkeypatch):
    """A regime missing a required slot is rejected."""

    def incomplete():
        """Return a prior bag missing the I0 slot.

        Returns
        -------
        dict
            Prior bag without the required ``I0`` entry.
        """
        bag = run_prior_regimes.example_priors()
        del bag["I0"]
        return bag

    monkeypatch.setitem(run_prior_regimes.REGIMES, "broken", incomplete)
    with pytest.raises(ValueError, match="missing prior slots"):
        run_prior_regimes._validate_regimes()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
