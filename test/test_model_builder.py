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
    """
    Create a configured builder (no population structure at configure time).

    Returns
    -------
    ModelBuilder
        Configured model builder.
    """
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
        name="hospital",
        ascertainment_rate_rv=DeterministicVariable("ihr", 0.01),
        delay_distribution_rv=delay,
        noise=NegativeBinomialNoise(DeterministicVariable("conc", 10.0)),
    )
    builder.add_observation(obs)

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

    def test_rejects_n_initialization_points_at_configure_time(self):
        """Test that n_initialization_points is rejected at configure time."""
        builder = ModelBuilder()
        gen_int = DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3]))

        with pytest.raises(ValueError, match="Do not specify n_initialization_points"):
            builder.configure_latent(
                HierarchicalInfections,
                gen_int_rv=gen_int,
                I0_rv=DeterministicVariable("I0", 0.001),
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_temporal=RandomWalk(),
                subpop_temporal=RandomWalk(),
                n_initialization_points=10,  # Should fail
            )

    def test_rejects_reconfiguring_latent(self):
        """Test that configuring latent twice raises RuntimeError."""
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

        with pytest.raises(RuntimeError, match="already configured"):
            builder.configure_latent(
                HierarchicalInfections,
                gen_int_rv=gen_int,
                I0_rv=DeterministicVariable("I0", 0.001),
                initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
                baseline_temporal=RandomWalk(),
                subpop_temporal=RandomWalk(),
            )

    def test_rejects_duplicate_observation_name(self, simple_builder):
        """Test that adding duplicate observation name raises ValueError."""
        delay = DeterministicPMF("delay2", jnp.array([0.5, 0.5]))
        obs = Counts(
            name="hospital",  # Same name as existing observation
            ascertainment_rate_rv=DeterministicVariable("ihr2", 0.02),
            delay_distribution_rv=delay,
            noise=NegativeBinomialNoise(DeterministicVariable("conc2", 20.0)),
        )

        with pytest.raises(ValueError, match="already added"):
            simple_builder.add_observation(obs)

    def test_build_creates_model(self, simple_builder):
        """Test that build() creates a MultiSignalModel."""
        model = simple_builder.build()
        assert isinstance(model, MultiSignalModel)

    def test_build_without_latent_raises_error(self):
        """Test that build() without configure_latent raises ValueError."""
        builder = ModelBuilder()

        with pytest.raises(ValueError, match="Must call configure_latent"):
            builder.build()

    def test_compute_n_initialization_points_without_latent_raises(self):
        """Test that compute_n_initialization_points without latent raises."""
        builder = ModelBuilder()

        with pytest.raises(ValueError, match="Must call configure_latent"):
            builder.compute_n_initialization_points()

    def test_compute_n_initialization_points_without_gen_int_raises(self):
        """Test that compute_n_initialization_points without gen_int_rv raises."""
        builder = ModelBuilder()
        # Configure latent without gen_int_rv
        builder.latent_class = HierarchicalInfections
        builder.latent_params = {}  # Missing gen_int_rv

        with pytest.raises(ValueError, match="gen_int_rv is required"):
            builder.compute_n_initialization_points()

    def test_compute_n_initialization_points_returns_correct_value(
        self, simple_builder
    ):
        """Test that compute_n_initialization_points returns max of lookbacks."""
        n_init = simple_builder.compute_n_initialization_points()
        # gen_int has 3 elements, delay has 4 elements, so max is 4
        assert n_init == 4


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

    def test_validate_data_rejects_negative_times(self, simple_builder):
        """Test that negative times raises error."""
        model = simple_builder.build()

        with pytest.raises(ValueError, match="cannot contain negative"):
            model.validate_data(
                n_days_post_init=30,
                obs_fractions=OBS_FRACTIONS,
                unobs_fractions=UNOBS_FRACTIONS,
                hospital={
                    "obs": jnp.array([10]),
                    "times": jnp.array([-1]),
                },
            )

    def test_validate_data_rejects_unknown_observation(self, simple_builder):
        """Test that unknown observation name raises error."""
        model = simple_builder.build()

        with pytest.raises(ValueError, match="Unknown observation"):
            model.validate_data(
                n_days_post_init=30,
                obs_fractions=OBS_FRACTIONS,
                unobs_fractions=UNOBS_FRACTIONS,
                unknown_obs={
                    "obs": jnp.array([10]),
                    "times": jnp.array([5]),
                },
            )

    def test_validate_data_rejects_mismatched_obs_times_length(self, simple_builder):
        """Test that mismatched obs and times lengths raises error."""
        model = simple_builder.build()

        with pytest.raises(ValueError, match="obs length.*must match times length"):
            model.validate_data(
                n_days_post_init=30,
                obs_fractions=OBS_FRACTIONS,
                unobs_fractions=UNOBS_FRACTIONS,
                hospital={
                    "obs": jnp.array([10, 20, 30]),  # 3 elements
                    "times": jnp.array([5, 10]),  # 2 elements
                },
            )

    def test_validate_method_calls_internal_validate(self, simple_builder):
        """Test that validate() method calls _validate()."""
        model = simple_builder.build()
        # Should not raise
        model.validate()


class TestMultiSignalModelObservationValidation:
    """Test observation process validation in MultiSignalModel."""

    def test_rejects_observation_without_infection_resolution(self):
        """Test that observations must implement infection_resolution()."""
        from pyrenew.observation.base import BaseObservationProcess

        class BadObservation(BaseObservationProcess):
            """Observation that raises NotImplementedError for infection_resolution."""

            def __init__(self):
                """Initialize without temporal_pmf_rv."""
                self.name = "bad"
                self.temporal_pmf_rv = None

            def sample(self, **kwargs):
                """Sample stub."""
                pass

            def validate(self):
                """Validate stub."""
                pass

            def lookback_days(self):
                """Return lookback."""
                return 1

            def infection_resolution(self):
                """Raise NotImplementedError to simulate missing implementation."""
                raise NotImplementedError("Not implemented")

            def _predicted_obs(self, infections):
                """Predicted obs stub."""
                return infections

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

        bad_obs = BadObservation()
        builder.add_observation(bad_obs)

        with pytest.raises(ValueError, match="must implement infection_resolution"):
            builder.build()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
