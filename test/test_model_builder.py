"""
Tests for ModelBuilder and MultiSignalModel.
"""

import jax.numpy as jnp
import numpyro
import pytest
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable

from pyrenew.latent import HierarchicalInfections, RandomWalk
from pyrenew.model import ModelBuilder, MultiSignalModel
from pyrenew.observation import Counts, NegativeBinomialNoise

# Standard population structure for tests
OBS_FRACTIONS = jnp.array([0.3, 0.25])
UNOBS_FRACTIONS = jnp.array([0.45])


@pytest.fixture
def simple_builder():
    """Create a configured builder (no population structure at configure time)."""
    builder = ModelBuilder()
    gen_int = DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3]))

    builder.configure_latent(
        HierarchicalInfections,
        gen_int_rv=gen_int,
        I0_rv=DeterministicVariable("I0", 0.001),
        initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
        baseline_temporal=RandomWalk(),
        subpop_temporal=RandomWalk(),
    )

    delay = DeterministicPMF("delay", jnp.array([0.1, 0.3, 0.4, 0.2]))
    obs = Counts(
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
        delay_distribution_rv=delay,
        noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
    )
    builder.add_observation(obs, "hospital")

    return builder


class TestModelBuilderConfiguration:
    """Test ModelBuilder configuration."""

    def test_rejects_population_structure_at_configure_time(self):
        """Test that population structure params are rejected at configure time."""
        builder = ModelBuilder()
        gen_int = DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3]))

        with pytest.raises(ValueError, match="Do not specify"):
            builder.configure_latent(
                HierarchicalInfections,
                gen_int_rv=gen_int,
                I0_rv=DeterministicVariable("I0", 0.001),
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_temporal=RandomWalk(),
                subpop_temporal=RandomWalk(),
                obs_fractions=jnp.array([0.5, 0.5]),  # Should fail
            )

    def test_build_creates_model(self, simple_builder):
        """Test that build() creates a MultiSignalModel."""
        model = simple_builder.build()
        assert isinstance(model, MultiSignalModel)


class TestMultiSignalModelSampling:
    """Test MultiSignalModel sampling with population structure at sample time."""

    def test_sample_with_population_structure(self, simple_builder):
        """Test that sample() works with population structure at sample time."""
        model = simple_builder.build()
        n_days = 30
        n_total = model.latent.n_initialization_points + n_days

        with numpyro.handlers.seed(rng_seed=42):
            inf_aggregate, inf_all, inf_obs, inf_unobs = model.sample(
                n_days_post_init=n_days,
                population_size=1_000_000,
                obs_fractions=OBS_FRACTIONS,
                unobs_fractions=UNOBS_FRACTIONS,
                hospital={"obs": None},
            )

        assert inf_aggregate.shape == (n_total,)
        assert inf_all.shape == (n_total, 3)  # K=3
        assert inf_obs.shape == (n_total, 2)  # K_obs=2

    def test_fit_with_population_structure(self, simple_builder):
        """Test that fit() works with population structure at sample time."""
        model = simple_builder.build()
        n_days = 10
        n_total = model.latent.n_initialization_points + n_days

        times = jnp.array([5, 6, 7, 8, 9])
        obs = jnp.array([10, 12, 15, 14, 11])

        mcmc = model.fit(
            n_days_post_init=n_days,
            population_size=1_000_000,
            obs_fractions=OBS_FRACTIONS,
            unobs_fractions=UNOBS_FRACTIONS,
            num_warmup=5,
            num_samples=5,
            hospital={"obs": obs, "times": times},
        )

        samples = mcmc.get_samples()
        assert "latent_infections" in samples
        assert samples["latent_infections"].shape == (5, n_total)


class TestMultiSignalModelValidation:
    """Test data validation."""

    def test_validate_data_requires_population_structure(self, simple_builder):
        """Test that validate_data requires population structure."""
        model = simple_builder.build()

        # Should work with population structure
        model.validate_data(
            n_days_post_init=30,
            obs_fractions=OBS_FRACTIONS,
            unobs_fractions=UNOBS_FRACTIONS,
            hospital={
                "obs": jnp.array([10, 20]),
                "times": jnp.array([5, 10]),
            },
        )

    def test_validate_data_rejects_out_of_bounds_times(self, simple_builder):
        """Test that times exceeding n_total_days raises error."""
        model = simple_builder.build()
        n_total = model.latent.n_initialization_points + 30

        with pytest.raises(ValueError, match="times contains"):
            model.validate_data(
                n_days_post_init=30,
                obs_fractions=OBS_FRACTIONS,
                unobs_fractions=UNOBS_FRACTIONS,
                hospital={
                    "obs": jnp.array([10]),
                    "times": jnp.array([n_total + 10]),
                },
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
