"""
Unit tests for PartitionedInfections.
"""

import jax.numpy as jnp
import numpyro
import pytest

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import PartitionedInfections, RandomWalk


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
    Create a PartitionedInfections with the new keyword-only API.

    Returns
    -------
    PartitionedInfections
        Configured infection process.
    """
    return PartitionedInfections(
        gen_int_rv=gen_int_rv,
        I0_rv=DeterministicVariable("I0", 0.001),
        initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
        rt_temporal=RandomWalk(),
        allocation_temporal=RandomWalk(),
    )


class TestPartitionedInfectionsSample:
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

        # Jurisdiction B: 4 observed, 1 unobserved
        with numpyro.handlers.seed(rng_seed=42):
            _, inf_all_b, _, _ = process.sample(
                n_days_post_init=30,
                obs_fractions=jnp.array([0.15, 0.20, 0.25, 0.10]),
                unobs_fractions=jnp.array([0.30]),
            )

        assert inf_all_a.shape[1] == 3  # K=3
        assert inf_all_b.shape[1] == 5  # K=5

    def test_allocated_infections_sum_to_total(self, process):
        """Test that allocated infections sum to jurisdiction total."""
        with numpyro.handlers.seed(rng_seed=42):
            inf_juris, inf_all, _, _ = process.sample(
                n_days_post_init=30,
                obs_fractions=jnp.array([0.3, 0.25]),
                unobs_fractions=jnp.array([0.45]),
            )

        infections_sum = jnp.sum(inf_all, axis=1)
        assert jnp.allclose(inf_juris, infections_sum, atol=1e-6)

    def test_allocation_proportions_sum_to_one(self, process):
        """Test that allocation proportions sum to 1 at each timepoint."""
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                process.sample(
                    n_days_post_init=30,
                    obs_fractions=jnp.array([0.3, 0.25]),
                    unobs_fractions=jnp.array([0.45]),
                )

        proportions = trace["allocation_proportions"]["value"]
        proportion_sums = jnp.sum(proportions, axis=1)

        assert jnp.allclose(proportion_sums, 1.0, atol=1e-6)


class TestPartitionedInfectionsValidation:
    """Test validation of inputs."""

    def test_rejects_missing_I0_rv(self, gen_int_rv):
        """Test that None I0_rv is rejected."""
        with pytest.raises(ValueError, match="I0_rv is required"):
            PartitionedInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=None,
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                rt_temporal=RandomWalk(),
                allocation_temporal=RandomWalk(),
            )

    def test_rejects_missing_initial_log_rt_rv(self, gen_int_rv):
        """Test that None initial_log_rt_rv is rejected."""
        with pytest.raises(ValueError, match="initial_log_rt_rv is required"):
            PartitionedInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                initial_log_rt_rv=None,
                rt_temporal=RandomWalk(),
                allocation_temporal=RandomWalk(),
            )

    def test_rejects_missing_rt_temporal(self, gen_int_rv):
        """Test that None rt_temporal is rejected."""
        with pytest.raises(ValueError, match="rt_temporal is required"):
            PartitionedInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                rt_temporal=None,
                allocation_temporal=RandomWalk(),
            )

    def test_rejects_missing_allocation_temporal(self, gen_int_rv):
        """Test that None allocation_temporal is rejected."""
        with pytest.raises(ValueError, match="allocation_temporal is required"):
            PartitionedInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                rt_temporal=RandomWalk(),
                allocation_temporal=None,
            )

    def test_rejects_invalid_I0(self, gen_int_rv):
        """Test that invalid I0 values are rejected."""
        with pytest.raises(ValueError, match="I0 must be positive"):
            PartitionedInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", -0.1),
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                rt_temporal=RandomWalk(),
                allocation_temporal=RandomWalk(),
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


class TestPartitionedInfectionsSingleSubpop:
    """Test partitioned infections with single subpopulation (K=1)."""

    def test_single_subpop_sample(self, gen_int_rv):
        """Test sampling with a single observed subpopulation."""
        process = PartitionedInfections(
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", 0.001),
            initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
            rt_temporal=RandomWalk(),
            allocation_temporal=RandomWalk(),
        )

        with numpyro.handlers.seed(rng_seed=42):
            result = process.sample(
                n_days_post_init=30,
                obs_fractions=jnp.array([1.0]),
                unobs_fractions=jnp.array([]),
            )

        inf_juris, inf_all, inf_obs, inf_unobs = result
        n_total = process.n_initialization_points + 30

        # K=1 case: no allocation deviations needed
        assert inf_juris.shape == (n_total,)
        assert inf_all.shape == (n_total, 1)
        assert inf_obs.shape == (n_total, 1)
        assert inf_unobs.shape == (n_total, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
