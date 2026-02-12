"""
Unit tests for HierarchicalInfections.
"""

import jax.numpy as jnp
import numpyro
import pytest

from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import HierarchicalInfections, RandomWalk


class TestHierarchicalInfectionsSample:
    """Test sample method with population structure at sample time."""

    def test_jurisdiction_total_is_weighted_sum(self, hierarchical_infections):
        """Test that jurisdiction total equals weighted sum of subpopulations."""
        fractions = jnp.array([0.3, 0.25, 0.45])

        with numpyro.handlers.seed(rng_seed=42):
            inf_juris, inf_all = hierarchical_infections.sample(
                n_days_post_init=30,
                subpop_fractions=fractions,
            )

        expected = jnp.sum(inf_all * fractions[jnp.newaxis, :], axis=1)

        assert jnp.allclose(inf_juris, expected, atol=1e-6)

    def test_deviations_sum_to_zero(self, hierarchical_infections):
        """Test that subpopulation deviations sum to zero (identifiability)."""
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                hierarchical_infections.sample(
                    n_days_post_init=30,
                    subpop_fractions=jnp.array([0.3, 0.25, 0.45]),
                )

        deviations = trace["latent_infections/subpop_deviations"]["value"]
        deviation_sums = jnp.sum(deviations, axis=1)

        assert jnp.allclose(deviation_sums, 0.0, atol=1e-6)

    def test_infections_are_positive(self, hierarchical_infections):
        """Test that all infections are positive (epidemiological invariant)."""
        with numpyro.handlers.seed(rng_seed=42):
            inf_juris, inf_all = hierarchical_infections.sample(
                n_days_post_init=30,
                subpop_fractions=jnp.array([0.3, 0.25, 0.45]),
            )

        assert jnp.all(inf_juris > 0)
        assert jnp.all(inf_all > 0)

    @pytest.mark.parametrize(
        "fractions",
        [
            jnp.array([1.0]),
            jnp.array([0.3, 0.25, 0.45]),
            jnp.array([0.10, 0.14, 0.21, 0.22, 0.07, 0.26]),
        ],
        ids=["K=1", "K=3", "K=6"],
    )
    def test_shape_and_positivity_across_subpop_counts(
        self, hierarchical_infections, fractions
    ):
        """Test correct shapes and positivity for varying numbers of subpops."""
        n_days_post_init = 30
        n_total = hierarchical_infections.n_initialization_points + n_days_post_init
        n_subpops = len(fractions)

        with numpyro.handlers.seed(rng_seed=42):
            inf_juris, inf_all = hierarchical_infections.sample(
                n_days_post_init=n_days_post_init,
                subpop_fractions=fractions,
            )

        assert inf_juris.shape == (n_total,)
        assert inf_all.shape == (n_total, n_subpops)
        assert jnp.all(inf_juris > 0)
        assert jnp.all(inf_all > 0)

        # Weighted sum property
        expected = jnp.sum(inf_all * fractions[jnp.newaxis, :], axis=1)
        assert jnp.allclose(inf_juris, expected, atol=1e-6)


class TestHierarchicalInfectionsValidation:
    """Test validation of inputs."""

    def test_rejects_missing_I0_rv(self, gen_int_rv):
        """Test that None I0_rv is rejected."""
        with pytest.raises(ValueError, match="I0_rv is required"):
            HierarchicalInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=None,
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_rt_process=RandomWalk(),
                subpop_rt_deviation_process=RandomWalk(),
                n_initialization_points=7,
            )

    def test_rejects_missing_initial_log_rt_rv(self, gen_int_rv):
        """Test that None initial_log_rt_rv is rejected."""
        with pytest.raises(ValueError, match="initial_log_rt_rv is required"):
            HierarchicalInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                initial_log_rt_rv=None,
                baseline_rt_process=RandomWalk(),
                subpop_rt_deviation_process=RandomWalk(),
                n_initialization_points=7,
            )

    def test_rejects_missing_baseline_rt_process(self, gen_int_rv):
        """Test that None baseline_rt_process is rejected."""
        with pytest.raises(ValueError, match="baseline_rt_process is required"):
            HierarchicalInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_rt_process=None,
                subpop_rt_deviation_process=RandomWalk(),
                n_initialization_points=7,
            )

    def test_rejects_missing_subpop_rt_deviation_process(self, gen_int_rv):
        """Test that None subpop_rt_deviation_process is rejected."""
        with pytest.raises(ValueError, match="subpop_rt_deviation_process is required"):
            HierarchicalInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_rt_process=RandomWalk(),
                subpop_rt_deviation_process=None,
                n_initialization_points=7,
            )

    def test_rejects_invalid_I0(self, gen_int_rv):
        """Test that invalid I0 values are rejected."""
        with pytest.raises(ValueError, match="I0 must be positive"):
            HierarchicalInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", -0.1),
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_rt_process=RandomWalk(),
                subpop_rt_deviation_process=RandomWalk(),
                n_initialization_points=7,
            )

    def test_rejects_insufficient_n_initialization_points(self, gen_int_rv):
        """Test that n_initialization_points < gen_int length is rejected."""
        with pytest.raises(
            ValueError, match="n_initialization_points must be at least"
        ):
            HierarchicalInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_rt_process=RandomWalk(),
                subpop_rt_deviation_process=RandomWalk(),
                n_initialization_points=2,
            )

    def test_rejects_fractions_not_summing_to_one(self, hierarchical_infections):
        """Test that fractions not summing to 1 raises error at sample time."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            with numpyro.handlers.seed(rng_seed=42):
                hierarchical_infections.sample(
                    n_days_post_init=30,
                    subpop_fractions=jnp.array([0.3, 0.25, 0.40]),
                )


class TestHierarchicalInfectionsPerSubpopI0:
    """Test per-subpopulation I0 values."""

    def test_per_subpop_I0_array(self, gen_int_rv):
        """Test with per-subpopulation I0 values and verify positivity."""
        process = HierarchicalInfections(
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", jnp.array([0.001, 0.002, 0.0015])),
            initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
            baseline_rt_process=RandomWalk(),
            subpop_rt_deviation_process=RandomWalk(),
            n_initialization_points=7,
        )

        with numpyro.handlers.seed(rng_seed=42):
            inf_juris, inf_all = process.sample(
                n_days_post_init=30,
                subpop_fractions=jnp.array([0.3, 0.25, 0.45]),
            )

        n_total = process.n_initialization_points + 30

        assert inf_juris.shape == (n_total,)
        assert inf_all.shape == (n_total, 3)
        assert jnp.all(inf_juris > 0)
        assert jnp.all(inf_all > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
