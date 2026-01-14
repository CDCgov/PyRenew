"""Unit tests for hierarchical prior distributions."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from pyrenew.deterministic import DeterministicVariable
from pyrenew.randomvariable import DistributionalVariable

from pyrenew.latent import (
    GammaGroupSdPrior,
    HierarchicalNormalPrior,
    StudentTGroupModePrior,
)


class TestHierarchicalNormalPrior:
    """Test HierarchicalNormalPrior."""

    def test_sample_shape(self):
        """Test that sample returns correct shape."""
        prior = HierarchicalNormalPrior(
            "effect", sd_rv=DeterministicVariable("sd", 1.0)
        )

        with numpyro.handlers.seed(rng_seed=42):
            samples = prior.sample(n_groups=5)

        assert samples.shape == (5,)

    def test_smaller_sd_produces_tighter_distribution(self):
        """Test that smaller sd produces samples closer to zero."""
        prior_tight = HierarchicalNormalPrior(
            "a", sd_rv=DeterministicVariable("sd_tight", 0.1)
        )
        prior_wide = HierarchicalNormalPrior(
            "b", sd_rv=DeterministicVariable("sd_wide", 10.0)
        )

        n_samples = 1000
        with numpyro.handlers.seed(rng_seed=42):
            samples_tight = prior_tight.sample(n_groups=n_samples)
        with numpyro.handlers.seed(rng_seed=43):
            samples_wide = prior_wide.sample(n_groups=n_samples)

        # Tight prior should have smaller standard deviation
        assert jnp.std(samples_tight) < jnp.std(samples_wide)

    def test_rejects_non_random_variable_sd(self):
        """Test that non-RandomVariable sd_rv is rejected."""
        with pytest.raises(TypeError, match="sd_rv must be a RandomVariable"):
            HierarchicalNormalPrior("effect", sd_rv=1.0)

    def test_accepts_distributional_variable_for_sd(self):
        """Test that DistributionalVariable can be used for sd_rv."""
        sd_rv = DistributionalVariable("sd", dist.HalfNormal(1.0))
        prior = HierarchicalNormalPrior("effect", sd_rv=sd_rv)

        with numpyro.handlers.seed(rng_seed=42):
            samples = prior.sample(n_groups=5)

        assert samples.shape == (5,)


class TestGammaGroupSdPrior:
    """Test GammaGroupSdPrior."""

    def test_sample_shape(self):
        """Test that sample returns correct shape."""
        prior = GammaGroupSdPrior(
            "sd",
            sd_mean_rv=DeterministicVariable("sd_mean", 0.5),
            sd_concentration_rv=DeterministicVariable("sd_conc", 4.0),
        )

        with numpyro.handlers.seed(rng_seed=42):
            samples = prior.sample(n_groups=5)

        assert samples.shape == (5,)

    def test_respects_sd_min(self):
        """Test that sd_min is enforced as lower bound."""
        prior = GammaGroupSdPrior(
            "sd",
            sd_mean_rv=DeterministicVariable("sd_mean", 0.1),
            sd_concentration_rv=DeterministicVariable("sd_conc", 4.0),
            sd_min=0.5,
        )

        with numpyro.handlers.seed(rng_seed=42):
            samples = prior.sample(n_groups=100)

        assert jnp.all(samples >= 0.5)

    def test_rejects_non_random_variable_params(self):
        """Test that non-RandomVariable parameters are rejected."""
        with pytest.raises(TypeError, match="sd_mean_rv must be a RandomVariable"):
            GammaGroupSdPrior(
                "sd",
                sd_mean_rv=0.5,
                sd_concentration_rv=DeterministicVariable("sd_conc", 4.0),
            )

        with pytest.raises(
            TypeError, match="sd_concentration_rv must be a RandomVariable"
        ):
            GammaGroupSdPrior(
                "sd",
                sd_mean_rv=DeterministicVariable("sd_mean", 0.5),
                sd_concentration_rv=4.0,
            )

    def test_rejects_negative_sd_min(self):
        """Test that negative sd_min is rejected."""
        with pytest.raises(ValueError, match="sd_min must be non-negative"):
            GammaGroupSdPrior(
                "sd",
                sd_mean_rv=DeterministicVariable("sd_mean", 0.5),
                sd_concentration_rv=DeterministicVariable("sd_conc", 4.0),
                sd_min=-0.1,
            )


class TestStudentTGroupModePrior:
    """Test StudentTGroupModePrior."""

    def test_sample_shape(self):
        """Test that sample returns correct shape."""
        prior = StudentTGroupModePrior(
            "mode",
            sd_rv=DeterministicVariable("sd", 1.0),
            df_rv=DeterministicVariable("df", 4.0),
        )

        with numpyro.handlers.seed(rng_seed=42):
            samples = prior.sample(n_groups=5)

        assert samples.shape == (5,)

    def test_heavier_tails_than_normal(self):
        """Test Student-t produces more extreme values than Normal."""
        # df=2 gives very heavy tails
        student_prior = StudentTGroupModePrior(
            "s",
            sd_rv=DeterministicVariable("sd_s", 1.0),
            df_rv=DeterministicVariable("df", 2.0),
        )
        normal_prior = HierarchicalNormalPrior(
            "n", sd_rv=DeterministicVariable("sd_n", 1.0)
        )

        n_samples = 5000
        with numpyro.handlers.seed(rng_seed=42):
            student_samples = student_prior.sample(n_groups=n_samples)
        with numpyro.handlers.seed(rng_seed=42):
            normal_samples = normal_prior.sample(n_groups=n_samples)

        # Student-t should have more extreme values (higher max absolute value)
        assert jnp.max(jnp.abs(student_samples)) > jnp.max(jnp.abs(normal_samples))

    def test_rejects_non_random_variable_params(self):
        """Test that non-RandomVariable parameters are rejected."""
        with pytest.raises(TypeError, match="sd_rv must be a RandomVariable"):
            StudentTGroupModePrior(
                "mode",
                sd_rv=1.0,
                df_rv=DeterministicVariable("df", 4.0),
            )

        with pytest.raises(TypeError, match="df_rv must be a RandomVariable"):
            StudentTGroupModePrior(
                "mode",
                sd_rv=DeterministicVariable("sd", 1.0),
                df_rv=4.0,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
