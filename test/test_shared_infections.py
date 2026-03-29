"""
Unit tests for SharedInfections.
"""

import jax.numpy as jnp
import numpyro
import pytest

from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import AR1, RandomWalk
from pyrenew.latent.shared_infections import SharedInfections


@pytest.fixture
def shared_infections(gen_int_rv):
    """
    Pre-configured SharedInfections instance.

    Returns
    -------
    SharedInfections
        Configured infection process with realistic parameters.
    """
    return SharedInfections(
        gen_int_rv=gen_int_rv,
        I0_rv=DeterministicVariable("I0", 0.001),
        initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
        shared_rt_process=AR1(autoreg=0.9, innovation_sd=0.05),
        n_initialization_points=7,
    )


class TestSharedInfectionsSample:
    """Test sample method for SharedInfections."""

    def test_output_shapes(self, shared_infections):
        """Test that output shapes are correct for a single-population model."""
        n_days_post_init = 30
        n_total = shared_infections.n_initialization_points + n_days_post_init

        with numpyro.handlers.seed(rng_seed=42):
            inf_agg, inf_all = shared_infections.sample(
                n_days_post_init=n_days_post_init,
            )

        assert inf_agg.shape == (n_total,)
        assert inf_all.shape == (n_total, 1)

    def test_aggregate_equals_single_subpop(self, shared_infections):
        """Test that aggregate infections equal the single subpopulation column."""
        with numpyro.handlers.seed(rng_seed=42):
            inf_agg, inf_all = shared_infections.sample(
                n_days_post_init=30,
            )

        assert jnp.allclose(inf_agg, inf_all[:, 0], atol=1e-6)

    def test_infections_are_positive(self, shared_infections):
        """Test that all infections are positive."""
        with numpyro.handlers.seed(rng_seed=42):
            inf_agg, inf_all = shared_infections.sample(
                n_days_post_init=30,
            )

        assert jnp.all(inf_agg > 0)
        assert jnp.all(inf_all > 0)

    def test_deterministic_sites_recorded(self, shared_infections):
        """Test that expected numpyro deterministic sites are recorded."""
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                shared_infections.sample(n_days_post_init=30)

        expected_sites = [
            "latent_infections::I0_init",
            "latent_infections::log_rt_shared",
            "latent_infections::rt_shared",
            "latent_infections::infections_aggregate",
        ]
        for site in expected_sites:
            assert site in trace, f"Missing deterministic site: {site}"

    def test_rt_is_exp_of_log_rt(self, shared_infections):
        """Test that recorded Rt equals exp of recorded log Rt."""
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                shared_infections.sample(n_days_post_init=30)

        log_rt = trace["latent_infections::log_rt_shared"]["value"]
        rt = trace["latent_infections::rt_shared"]["value"]

        assert jnp.allclose(rt, jnp.exp(log_rt), atol=1e-6)

    def test_default_fractions_used_when_none(self, shared_infections):
        """Test that default fractions [1.0] are used when not provided."""
        with numpyro.handlers.seed(rng_seed=42):
            inf_agg, inf_all = shared_infections.sample(
                n_days_post_init=30,
                subpop_fractions=None,
            )

        assert inf_all.shape[1] == 1

    def test_explicit_fractions_one(self, shared_infections):
        """Test that explicit fractions [1.0] produce same results as default."""
        with numpyro.handlers.seed(rng_seed=42):
            inf_agg_default, inf_all_default = shared_infections.sample(
                n_days_post_init=30,
            )

        with numpyro.handlers.seed(rng_seed=42):
            inf_agg_explicit, inf_all_explicit = shared_infections.sample(
                n_days_post_init=30,
                subpop_fractions=jnp.array([1.0]),
            )

        assert jnp.allclose(inf_agg_default, inf_agg_explicit, atol=1e-6)
        assert jnp.allclose(inf_all_default, inf_all_explicit, atol=1e-6)

    def test_different_seeds_give_different_results(self, shared_infections):
        """Test that different RNG seeds produce different trajectories."""
        with numpyro.handlers.seed(rng_seed=1):
            inf_agg_1, _ = shared_infections.sample(n_days_post_init=30)

        with numpyro.handlers.seed(rng_seed=999):
            inf_agg_2, _ = shared_infections.sample(n_days_post_init=30)

        assert not jnp.allclose(inf_agg_1, inf_agg_2)

    def test_custom_name_prefix(self, gen_int_rv):
        """Test that custom name prefix is used in deterministic sites."""
        process = SharedInfections(
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", 0.001),
            initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
            shared_rt_process=RandomWalk(),
            n_initialization_points=7,
            name="my_infections",
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                process.sample(n_days_post_init=30)

        assert "my_infections::rt_shared" in trace


class TestSharedInfectionsValidation:
    """Test validation of inputs."""

    def test_rejects_missing_I0_rv(self, gen_int_rv):
        """Test that None I0_rv is rejected."""
        with pytest.raises(ValueError, match="I0_rv is required"):
            SharedInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=None,
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                shared_rt_process=RandomWalk(),
                n_initialization_points=7,
            )

    def test_rejects_missing_initial_log_rt_rv(self, gen_int_rv):
        """Test that None initial_log_rt_rv is rejected."""
        with pytest.raises(ValueError, match="initial_log_rt_rv is required"):
            SharedInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                initial_log_rt_rv=None,
                shared_rt_process=RandomWalk(),
                n_initialization_points=7,
            )

    def test_rejects_missing_shared_rt_process(self, gen_int_rv):
        """Test that None shared_rt_process is rejected."""
        with pytest.raises(ValueError, match="shared_rt_process is required"):
            SharedInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                shared_rt_process=None,
                n_initialization_points=7,
            )

    def test_rejects_invalid_I0(self, gen_int_rv):
        """Test that invalid I0 values are rejected at construction."""
        with pytest.raises(ValueError, match="I0 must be positive"):
            SharedInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", -0.1),
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                shared_rt_process=RandomWalk(),
                n_initialization_points=7,
            )

    def test_rejects_I0_greater_than_one(self, gen_int_rv):
        """Test that I0 > 1 is rejected at construction."""
        with pytest.raises(ValueError, match="I0 must be <= 1"):
            SharedInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 1.5),
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                shared_rt_process=RandomWalk(),
                n_initialization_points=7,
            )

    def test_rejects_insufficient_n_initialization_points(self, gen_int_rv):
        """Test that n_initialization_points < gen_int length is rejected."""
        with pytest.raises(
            ValueError, match="n_initialization_points must be at least"
        ):
            SharedInfections(
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                shared_rt_process=RandomWalk(),
                n_initialization_points=2,
            )

    def test_rejects_fractions_not_summing_to_one(self, shared_infections):
        """Test that fractions not summing to 1 raises error at sample time."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            with numpyro.handlers.seed(rng_seed=42):
                shared_infections.sample(
                    n_days_post_init=30,
                    subpop_fractions=jnp.array([0.5]),
                )

    def test_rejects_multiple_subpop_fractions_even_if_sum_to_one(
        self, shared_infections
    ):
        """Test that multi-element fractions are rejected for shared infections."""
        with pytest.raises(
            ValueError,
            match="requires exactly one subpopulation with fraction \\[1.0\\]",
        ):
            with numpyro.handlers.seed(rng_seed=42):
                shared_infections.sample(
                    n_days_post_init=30,
                    subpop_fractions=jnp.array([0.5, 0.5]),
                )

    def test_rejects_non_scalar_I0(self, gen_int_rv):
        """Test that vector-valued I0 is rejected with a clear error."""
        process = SharedInfections(
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", jnp.array([0.001, 0.002])),
            initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
            shared_rt_process=RandomWalk(),
            n_initialization_points=7,
        )

        with pytest.raises(
            ValueError,
            match="requires I0_rv to return a scalar prevalence",
        ):
            with numpyro.handlers.seed(rng_seed=42):
                process.sample(n_days_post_init=30)

    def test_validate_passes(self, shared_infections):
        """Test that validate() succeeds for a properly constructed instance."""
        shared_infections.validate()

    def test_default_subpop_fractions(self, shared_infections):
        """Test that default_subpop_fractions returns [1.0]."""
        fracs = shared_infections.default_subpop_fractions()
        assert jnp.allclose(fracs, jnp.array([1.0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
