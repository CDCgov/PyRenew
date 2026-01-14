"""
Unit tests for temporal processes innovation_sd behavior.
"""

import jax.numpy as jnp
import numpyro
import pytest

from pyrenew.latent import AR1, DifferencedAR1, RandomWalk


class TestTemporalProcessInnovationSD:
    """Test that temporal processes correctly use innovation_sd parameter."""

    def test_random_walk_smaller_innovation_sd_produces_smoother_trajectory(
        self,
    ):
        """Verify that smaller innovation_sd produces less volatile trajectories."""
        n_timepoints = 100

        with numpyro.handlers.seed(rng_seed=42):
            rw_small = RandomWalk(innovation_sd=0.1)
            trajectory_small = rw_small.sample(n_timepoints=n_timepoints)

        with numpyro.handlers.seed(rng_seed=42):
            rw_large = RandomWalk(innovation_sd=1.0)
            trajectory_large = rw_large.sample(n_timepoints=n_timepoints)

        # Smaller innovation_sd should produce smaller step sizes
        steps_small = jnp.abs(jnp.diff(trajectory_small))
        steps_large = jnp.abs(jnp.diff(trajectory_large))

        assert jnp.mean(steps_small) < jnp.mean(steps_large)
        assert jnp.max(steps_small) < jnp.max(steps_large)

    def test_ar1_smaller_innovation_sd_produces_lower_variance(self):
        """Verify AR1 with smaller innovation_sd produces lower variance trajectories."""
        n_timepoints = 100
        autoreg = 0.7

        with numpyro.handlers.seed(rng_seed=42):
            ar_small = AR1(autoreg=autoreg, innovation_sd=0.2)
            trajectory_small = ar_small.sample(n_timepoints=n_timepoints)

        with numpyro.handlers.seed(rng_seed=42):
            ar_large = AR1(autoreg=autoreg, innovation_sd=1.0)
            trajectory_large = ar_large.sample(n_timepoints=n_timepoints)

        # After burn-in, smaller innovation_sd should have lower variance
        burn_in = 20
        var_small = jnp.var(trajectory_small[burn_in:])
        var_large = jnp.var(trajectory_large[burn_in:])

        assert var_small < var_large

    def test_differenced_ar1_smaller_innovation_sd_produces_smoother_changes(
        self,
    ):
        """Verify DifferencedAR1 with smaller innovation_sd produces smoother changes."""
        n_timepoints = 100
        autoreg = 0.6

        with numpyro.handlers.seed(rng_seed=42):
            dar_small = DifferencedAR1(autoreg=autoreg, innovation_sd=0.15)
            trajectory_small = dar_small.sample(n_timepoints=n_timepoints)

        with numpyro.handlers.seed(rng_seed=42):
            dar_large = DifferencedAR1(autoreg=autoreg, innovation_sd=0.8)
            trajectory_large = dar_large.sample(n_timepoints=n_timepoints)

        # Growth rates (differences) should have lower variance
        diffs_small = jnp.diff(trajectory_small)
        diffs_large = jnp.diff(trajectory_large)

        assert jnp.std(diffs_small) < jnp.std(diffs_large)

    def test_explicit_innovation_sd_overrides_constructor_value(self):
        """Verify that explicit innovation_sd in sample() overrides constructor value."""
        rw = RandomWalk(innovation_sd=0.1)

        with numpyro.handlers.seed(rng_seed=42):
            traj_constructor = rw.sample(n_timepoints=50)

        with numpyro.handlers.seed(rng_seed=42):
            traj_override = rw.sample(n_timepoints=50, innovation_sd=1.0)

        # Trajectories should be substantially different
        assert not jnp.allclose(traj_constructor, traj_override, rtol=0.1)

    def test_vectorized_sampling_respects_innovation_sd(self):
        """Verify vectorized sampling uses the innovation_sd parameter."""
        n_processes = 5
        n_timepoints = 50

        with numpyro.handlers.seed(rng_seed=42):
            rw_small = RandomWalk(innovation_sd=0.2)
            trajs_small = rw_small.sample(
                n_timepoints=n_timepoints, n_processes=n_processes
            )

        with numpyro.handlers.seed(rng_seed=42):
            rw_large = RandomWalk(innovation_sd=1.0)
            trajs_large = rw_large.sample(
                n_timepoints=n_timepoints, n_processes=n_processes
            )

        # Check that smaller innovation_sd produces smaller step sizes across all processes
        steps_small = jnp.abs(jnp.diff(trajs_small, axis=0))
        steps_large = jnp.abs(jnp.diff(trajs_large, axis=0))

        assert jnp.mean(steps_small) < jnp.mean(steps_large)

    def test_validation_rejects_non_positive_innovation_sd(self):
        """Verify that non-positive innovation_sd values are rejected."""
        with pytest.raises(ValueError, match="innovation_sd must be positive"):
            RandomWalk(innovation_sd=0.0)

        with pytest.raises(ValueError, match="innovation_sd must be positive"):
            RandomWalk(innovation_sd=-0.5)

        with pytest.raises(ValueError, match="innovation_sd must be positive"):
            AR1(autoreg=0.5, innovation_sd=-0.1)

        with pytest.raises(ValueError, match="innovation_sd must be positive"):
            DifferencedAR1(autoreg=0.5, innovation_sd=0.0)
