"""
Unit tests for PopulationInfections.
"""

import jax.numpy as jnp
import numpyro
import pytest

from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import AR1, RandomWalk, WeeklyTemporalProcess
from pyrenew.latent.population_infections import PopulationInfections
from pyrenew.time import MMWR_WEEK


class TestPopulationInfectionsSample:
    """Test sample method for PopulationInfections."""

    def test_output_shapes(self, population_infections):
        """Test that output shapes are correct for a single-population model."""
        n_days_post_init = 30
        n_total = population_infections.n_initialization_points + n_days_post_init

        with numpyro.handlers.seed(rng_seed=42):
            inf_agg, inf_all = population_infections.sample(
                n_days_post_init=n_days_post_init,
            )

        assert inf_agg.shape == (n_total,)
        assert inf_all.shape == (n_total, 1)

    def test_aggregate_equals_single_subpop(self, population_infections):
        """Test that aggregate infections equal the single subpopulation column."""
        with numpyro.handlers.seed(rng_seed=42):
            inf_agg, inf_all = population_infections.sample(
                n_days_post_init=30,
            )

        assert jnp.allclose(inf_agg, inf_all[:, 0], atol=1e-6)

    def test_infections_are_positive(self, population_infections):
        """Test that all infections are positive."""
        with numpyro.handlers.seed(rng_seed=42):
            inf_agg, inf_all = population_infections.sample(
                n_days_post_init=30,
            )

        assert jnp.all(inf_agg > 0)
        assert jnp.all(inf_all > 0)

    def test_deterministic_sites_recorded(self, population_infections):
        """Test that expected numpyro deterministic sites are recorded."""
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                population_infections.sample(n_days_post_init=30)

        expected_sites = [
            "population::I0_init",
            "population::log_rt_single",
            "population::rt_single",
            "population::infections_aggregate",
        ]
        for site in expected_sites:
            assert site in trace, f"Missing deterministic site: {site}"

    def test_rt_is_exp_of_log_rt(self, population_infections):
        """Test that recorded Rt equals exp of recorded log Rt."""
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                population_infections.sample(n_days_post_init=30)

        log_rt = trace["population::log_rt_single"]["value"]
        rt = trace["population::rt_single"]["value"]

        assert jnp.allclose(rt, jnp.exp(log_rt), atol=1e-6)

    def test_default_fractions_used_when_none(self, population_infections):
        """Test that default fractions [1.0] are used when not provided."""
        with numpyro.handlers.seed(rng_seed=42):
            inf_agg, inf_all = population_infections.sample(
                n_days_post_init=30,
                subpop_fractions=None,
            )

        assert inf_all.shape[1] == 1

    def test_explicit_fractions_one(self, population_infections):
        """Test that explicit fractions [1.0] produce same results as default."""
        with numpyro.handlers.seed(rng_seed=42):
            inf_agg_default, inf_all_default = population_infections.sample(
                n_days_post_init=30,
            )

        with numpyro.handlers.seed(rng_seed=42):
            inf_agg_explicit, inf_all_explicit = population_infections.sample(
                n_days_post_init=30,
                subpop_fractions=jnp.array([1.0]),
            )

        assert jnp.allclose(inf_agg_default, inf_agg_explicit, atol=1e-6)
        assert jnp.allclose(inf_all_default, inf_all_explicit, atol=1e-6)

    def test_different_seeds_give_different_results(self, population_infections):
        """Test that different RNG seeds produce different trajectories."""
        with numpyro.handlers.seed(rng_seed=1):
            inf_agg_1, _ = population_infections.sample(n_days_post_init=30)

        with numpyro.handlers.seed(rng_seed=999):
            inf_agg_2, _ = population_infections.sample(n_days_post_init=30)

        assert not jnp.allclose(inf_agg_1, inf_agg_2)

    def test_custom_name_prefix(self, gen_int_rv):
        """Test that custom name prefix is used in deterministic sites."""
        process = PopulationInfections(
            name="my_infections",
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", 0.001),
            log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
            single_rt_process=RandomWalk(),
            n_initialization_points=7,
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                process.sample(n_days_post_init=30)

        assert "my_infections::rt_single" in trace

    def test_first_day_dow_reaches_calendar_aligned_rt_process(self, gen_int_rv):
        """Weekly Rt receives model-axis day of week."""
        process = PopulationInfections(
            name="population",
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", 0.001),
            log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
            single_rt_process=WeeklyTemporalProcess(
                AR1(autoreg=0.9, innovation_sd=0.05),
                start_dow=MMWR_WEEK,
            ),
            n_initialization_points=7,
        )

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                process.sample(n_days_post_init=10, first_day_dow=3)

        log_rt = trace["population::log_rt_single"]["value"]
        weekly = trace["log_rt_single_weekly"]["value"]

        assert log_rt.shape == (17, 1)
        assert weekly.shape == (3, 1)
        assert jnp.allclose(log_rt[:3], log_rt[0])
        assert jnp.allclose(log_rt[3:10], log_rt[3])
        assert jnp.allclose(log_rt[10:17], log_rt[10])

    def test_wrong_rt_shape_raises(self, gen_int_rv, wrong_shape_temporal_process_cls):
        """Temporal processes must return daily-length single-process Rt."""
        process = PopulationInfections(
            name="population",
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", 0.001),
            log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
            single_rt_process=wrong_shape_temporal_process_cls(jnp.zeros((16, 2))),
            n_initialization_points=7,
        )

        with pytest.raises(ValueError, match="single_rt_process must return shape"):
            process.sample(n_days_post_init=10)


class TestPopulationInfectionsValidation:
    """Test validation of inputs."""

    def test_rejects_missing_I0_rv(self, gen_int_rv):
        """Test that None I0_rv is rejected."""
        with pytest.raises(ValueError, match="I0_rv is required"):
            PopulationInfections(
                name="population",
                gen_int_rv=gen_int_rv,
                I0_rv=None,
                log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
                single_rt_process=RandomWalk(),
                n_initialization_points=7,
            )

    def test_rejects_missing_log_rt_time_0_rv(self, gen_int_rv):
        """Test that None log_rt_time_0_rv is rejected."""
        with pytest.raises(ValueError, match="log_rt_time_0_rv is required"):
            PopulationInfections(
                name="population",
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                log_rt_time_0_rv=None,
                single_rt_process=RandomWalk(),
                n_initialization_points=7,
            )

    def test_rejects_missing_single_rt_process(self, gen_int_rv):
        """Test that None single_rt_process is rejected."""
        with pytest.raises(ValueError, match="single_rt_process is required"):
            PopulationInfections(
                name="population",
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
                single_rt_process=None,
                n_initialization_points=7,
            )

    def test_rejects_invalid_I0(self, gen_int_rv):
        """Test that invalid I0 values are rejected at construction."""
        with pytest.raises(ValueError, match="I0 must be positive"):
            PopulationInfections(
                name="population",
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", -0.1),
                log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
                single_rt_process=RandomWalk(),
                n_initialization_points=7,
            )

    def test_rejects_I0_greater_than_one(self, gen_int_rv):
        """Test that I0 > 1 is rejected at construction."""
        with pytest.raises(ValueError, match="I0 must be <= 1"):
            PopulationInfections(
                name="population",
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 1.5),
                log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
                single_rt_process=RandomWalk(),
                n_initialization_points=7,
            )

    def test_rejects_insufficient_n_initialization_points(self, gen_int_rv):
        """Test that n_initialization_points < gen_int length is rejected."""
        with pytest.raises(
            ValueError, match="n_initialization_points must be at least"
        ):
            PopulationInfections(
                name="population",
                gen_int_rv=gen_int_rv,
                I0_rv=DeterministicVariable("I0", 0.001),
                log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
                single_rt_process=RandomWalk(),
                n_initialization_points=2,
            )

    def test_rejects_fractions_not_summing_to_one(self, population_infections):
        """Test that fractions not summing to 1 raises error at sample time."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            with numpyro.handlers.seed(rng_seed=42):
                population_infections.sample(
                    n_days_post_init=30,
                    subpop_fractions=jnp.array([0.5]),
                )

    def test_rejects_multiple_subpop_fractions_even_if_sum_to_one(
        self, population_infections
    ):
        """Test that multi-element fractions are rejected for population infections."""
        with pytest.raises(
            ValueError,
            match="requires exactly one subpopulation with fraction \\[1.0\\]",
        ):
            with numpyro.handlers.seed(rng_seed=42):
                population_infections.sample(
                    n_days_post_init=30,
                    subpop_fractions=jnp.array([0.5, 0.5]),
                )

    def test_rejects_non_scalar_I0(self, gen_int_rv):
        """Test that vector-valued I0 is rejected with a clear error."""
        process = PopulationInfections(
            name="population",
            gen_int_rv=gen_int_rv,
            I0_rv=DeterministicVariable("I0", jnp.array([0.001, 0.002])),
            log_rt_time_0_rv=DeterministicVariable("log_rt_time_0", 0.0),
            single_rt_process=RandomWalk(),
            n_initialization_points=7,
        )

        with pytest.raises(
            ValueError,
            match="requires I0_rv to return a scalar prevalence",
        ):
            with numpyro.handlers.seed(rng_seed=42):
                process.sample(n_days_post_init=30)

    def test_validate_passes(self, population_infections):
        """Test that validate() succeeds for a properly constructed instance."""
        population_infections.validate()

    def test_default_subpop_fractions(self, population_infections):
        """Test that default_subpop_fractions returns [1.0]."""
        fracs = population_infections.default_subpop_fractions()
        assert jnp.allclose(fracs, jnp.array([1.0]))


class TestPopulationValidateAndPrepareI0:
    """Test _validate_and_prepare_I0 for PopulationInfections."""

    def test_accepts_valid_scalar(self, population_infections):
        """Test that a valid scalar I0 passes through unchanged."""
        pop = population_infections._parse_and_validate_fractions()
        I0 = jnp.array(0.01)
        result = population_infections._validate_and_prepare_I0(I0, pop)
        assert result.ndim == 0
        assert jnp.isclose(result, 0.01)

    def test_rejects_vector(self, population_infections):
        """Test that a vector I0 is rejected."""
        pop = population_infections._parse_and_validate_fractions()
        I0 = jnp.array([0.01, 0.02])
        with pytest.raises(ValueError, match="scalar prevalence"):
            population_infections._validate_and_prepare_I0(I0, pop)

    def test_rejects_negative(self, population_infections):
        """Test that negative I0 is rejected."""
        pop = population_infections._parse_and_validate_fractions()
        I0 = jnp.array(-0.01)
        with pytest.raises(ValueError, match="I0 must be positive"):
            population_infections._validate_and_prepare_I0(I0, pop)

    def test_rejects_greater_than_one(self, population_infections):
        """Test that I0 > 1 is rejected."""
        pop = population_infections._parse_and_validate_fractions()
        I0 = jnp.array(1.5)
        with pytest.raises(ValueError, match="I0 must be <= 1"):
            population_infections._validate_and_prepare_I0(I0, pop)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
