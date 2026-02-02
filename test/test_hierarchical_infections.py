"""
Unit tests for HierarchicalInfections.
"""

import jax.numpy as jnp
import numpyro
import pytest

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import HierarchicalInfections, RandomWalk


@pytest.fixture
def gen_int_rv():
    """
    Create a generation interval random variable.

    Returns
    -------
    DeterministicPMF
        Generation interval PMF.
    """
    return DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3]))


@pytest.fixture
def process(gen_int_rv):
    """
    Create a HierarchicalInfections with the new keyword-only API.

    Returns
    -------
    HierarchicalInfections
        Configured infection process.
    """
    return HierarchicalInfections(
        gen_int_rv=gen_int_rv,
        I0_rv=DeterministicVariable("I0", 0.001),
        initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
        baseline_temporal=RandomWalk(),
        subpop_temporal=RandomWalk(),
    )


class TestHierarchicalInfectionsSample:
    """Test sample method with population structure at sample time."""

    def test_sample_returns_correct_shapes(self, process):
        """Test that sample returns correct output shapes."""
        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                n_days_post_init=30,
                obs_fractions=jnp.array([0.3, 0.25]),
                unobs_fractions=jnp.array([0.45]),
            )

        inf_juris, inf_all, inf_obs, inf_unobs = result
        n_total = process.n_initialization_points + 30

        assert inf_juris.shape == (n_total,)
        assert inf_all.shape == (n_total, 3)
        assert inf_obs.shape == (n_total, 2)
        assert inf_unobs.shape == (n_total, 1)

    def test_same_model_different_jurisdictions(self, process):
        """Test that one model can fit different population structures."""
        # Jurisdiction A: 2 observed, 1 unobserved
        with numpyro.handlers.seed(rng_seed=42):
            _, inf_all_a, _, _ = process.sample(
                n_days_post_init=30,
                obs_fractions=jnp.array([0.3, 0.25]),
                unobs_fractions=jnp.array([0.45]),
            )

        # Jurisdiction B: 4 observed, 1 unobserved (different structure)
        with numpyro.handlers.seed(rng_seed=42):
            _, inf_all_b, _, _ = process.sample(
                n_days_post_init=30,
                obs_fractions=jnp.array([0.15, 0.20, 0.25, 0.10]),
                unobs_fractions=jnp.array([0.30]),
            )

        assert inf_all_a.shape[1] == 3  # K=3
        assert inf_all_b.shape[1] == 5  # K=5

    def test_jurisdiction_total_is_weighted_sum(self, process):
        """Test that jurisdiction total equals weighted sum of subpopulations."""
        obs_fracs = jnp.array([0.3, 0.25])
        unobs_fracs = jnp.array([0.45])

        with numpyro.handlers.seed(rng_seed=42):
            inf_juris, inf_all, _, _ = process.sample(
                n_days_post_init=30,
                obs_fractions=obs_fracs,
                unobs_fractions=unobs_fracs,
            )

        all_fractions = jnp.concatenate([obs_fracs, unobs_fracs])
        expected = jnp.sum(inf_all * all_fractions[jnp.newaxis, :], axis=1)

        assert jnp.allclose(inf_juris, expected, atol=1e-6)

    def test_deviations_sum_to_zero(self, process):
        """Test that subpopulation deviations sum to zero (identifiability)."""
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                process.sample(
                    n_days_post_init=30,
                    obs_fractions=jnp.array([0.3, 0.25]),
                    unobs_fractions=jnp.array([0.45]),
                )

        deviations = trace["subpop_deviations"]["value"]
        deviation_sums = jnp.sum(deviations, axis=1)

        assert jnp.allclose(deviation_sums, 0.0, atol=1e-6)


class TestHierarchicalInfectionsValidation:
    """Test validation of inputs."""

    def test_rejects_missing_I0_rv(self, gen_int_rv):
        """Test that None I0_rv is rejected."""
        with pytest.raises(ValueError, match="I0_rv is required"):
            HierarchicalInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=None,
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_temporal=RandomWalk(),
                subpop_temporal=RandomWalk(),
            )

    def test_rejects_missing_initial_log_rt_rv(self, gen_int_rv):
        """Test that None initial_log_rt_rv is rejected."""
        with pytest.raises(ValueError, match="initial_log_rt_rv is required"):
            HierarchicalInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                initial_log_rt_rv=None,
                baseline_temporal=RandomWalk(),
                subpop_temporal=RandomWalk(),
            )

    def test_rejects_missing_baseline_temporal(self, gen_int_rv):
        """Test that None baseline_temporal is rejected."""
        with pytest.raises(ValueError, match="baseline_temporal is required"):
            HierarchicalInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_temporal=None,
                subpop_temporal=RandomWalk(),
            )

    def test_rejects_missing_subpop_temporal(self, gen_int_rv):
        """Test that None subpop_temporal is rejected."""
        with pytest.raises(ValueError, match="subpop_temporal is required"):
            HierarchicalInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_temporal=RandomWalk(),
                subpop_temporal=None,
            )

    def test_rejects_invalid_I0(self, gen_int_rv):
        """Test that invalid I0 values are rejected."""
        with pytest.raises(ValueError, match="I0 must be positive"):
            HierarchicalInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", -0.1),
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_temporal=RandomWalk(),
                subpop_temporal=RandomWalk(),
            )

    def test_rejects_fractions_not_summing_to_one(self, process):
        """Test that fractions not summing to 1 raises error at sample time."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            with numpyro.handlers.seed(rng_seed=42):
                process.sample(
                    n_days_post_init=30,
                    obs_fractions=jnp.array([0.3, 0.25]),
                    unobs_fractions=jnp.array([0.40]),  # Sum is 0.95
                )

    def test_validate_method(self, process):
        """Test that validate() method runs without error."""
        process.validate()


class TestHierarchicalInfectionsPerSubpopI0:
    """Test per-subpopulation I0 values."""

    def test_per_subpop_I0_array(self, gen_int_rv):
        """Test with per-subpopulation I0 values (array instead of scalar)."""
        process = HierarchicalInfections(
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", jnp.array([0.001, 0.002, 0.0015])),
            initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
            baseline_temporal=RandomWalk(),
            subpop_temporal=RandomWalk(),
        )

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                n_days_post_init=30,
                obs_fractions=jnp.array([0.3, 0.25]),
                unobs_fractions=jnp.array([0.45]),
            )

        inf_juris, inf_all, _inf_obs, _inf_unobs = result
        n_total = process.n_initialization_points + 30

        assert inf_juris.shape == (n_total,)
        assert inf_all.shape == (n_total, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
