"""
Unit tests for temporal processes innovation_sd behavior.
"""

import jax.numpy as jnp
import numpyro
import pytest

from pyrenew.latent import AR1, DifferencedAR1, RandomWalk


class TestTemporalProcessRepr:
    """Test __repr__ methods for temporal processes."""

    def test_ar1_repr(self):
        """Test AR1 __repr__ method."""
        ar1 = AR1(autoreg=0.7, innovation_sd=0.5)
        repr_str = repr(ar1)
        assert "AR1" in repr_str
        assert "autoreg=0.7" in repr_str
        assert "innovation_sd=0.5" in repr_str

    def test_differenced_ar1_repr(self):
        """Test DifferencedAR1 __repr__ method."""
        dar1 = DifferencedAR1(autoreg=0.6, innovation_sd=0.3)
        repr_str = repr(dar1)
        assert "DifferencedAR1" in repr_str
        assert "autoreg=0.6" in repr_str
        assert "innovation_sd=0.3" in repr_str

    def test_random_walk_repr(self):
        """Test RandomWalk __repr__ method."""
        rw = RandomWalk(innovation_sd=0.2)
        repr_str = repr(rw)
        assert "RandomWalk" in repr_str
        assert "innovation_sd=0.2" in repr_str


class TestAR1VectorizedSampling:
    """Test AR1 vectorized sampling."""

    def test_ar1_vectorized_sample_shape(self):
        """Test that AR1 vectorized sample returns correct shape."""
        n_timepoints = 30
        n_processes = 5

        ar1 = AR1(autoreg=0.7, innovation_sd=0.3)

        with numpyro.handlers.seed(rng_seed=42):
            trajectories = ar1.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
            )

        assert trajectories.shape == (n_timepoints, n_processes)

    def test_ar1_vectorized_with_initial_values_array(self):
        """Test AR1 vectorized with array of initial values."""
        n_timepoints = 30
        n_processes = 4
        initial_values = jnp.array([0.0, 1.0, -1.0, 2.0])

        ar1 = AR1(autoreg=0.7, innovation_sd=0.3)

        with numpyro.handlers.seed(rng_seed=42):
            trajectories = ar1.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
                initial_value=initial_values,
            )

        assert trajectories.shape == (n_timepoints, n_processes)

    def test_ar1_vectorized_with_scalar_initial_value(self):
        """Test AR1 vectorized with scalar initial value (broadcast)."""
        n_timepoints = 30
        n_processes = 3

        ar1 = AR1(autoreg=0.7, innovation_sd=0.3)

        with numpyro.handlers.seed(rng_seed=42):
            trajectories = ar1.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
                initial_value=1.0,
            )

        assert trajectories.shape == (n_timepoints, n_processes)


class TestDifferencedAR1Sampling:
    """Test DifferencedAR1 sampling methods."""

    def test_differenced_ar1_single_sample_shape(self):
        """Test that DifferencedAR1 single sample returns correct shape."""
        n_timepoints = 30

        dar1 = DifferencedAR1(autoreg=0.6, innovation_sd=0.3)

        with numpyro.handlers.seed(rng_seed=42):
            trajectory = dar1.sample(n_timepoints=n_timepoints)

        assert trajectory.shape == (n_timepoints, 1)

    def test_differenced_ar1_single_with_initial_value(self):
        """Test DifferencedAR1 single sample with initial value."""
        n_timepoints = 30

        dar1 = DifferencedAR1(autoreg=0.6, innovation_sd=0.3)

        with numpyro.handlers.seed(rng_seed=42):
            trajectory = dar1.sample(
                n_timepoints=n_timepoints,
                initial_value=1.5,
            )

        assert trajectory.shape == (n_timepoints, 1)

    def test_differenced_ar1_vectorized_sample_shape(self):
        """Test that DifferencedAR1 vectorized sample returns correct shape."""
        n_timepoints = 30
        n_processes = 5

        dar1 = DifferencedAR1(autoreg=0.6, innovation_sd=0.3)

        with numpyro.handlers.seed(rng_seed=42):
            trajectories = dar1.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
            )

        assert trajectories.shape == (n_timepoints, n_processes)

    def test_differenced_ar1_vectorized_with_initial_values_array(self):
        """Test DifferencedAR1 vectorized with array of initial values."""
        n_timepoints = 30
        n_processes = 4
        initial_values = jnp.array([0.0, 1.0, -1.0, 2.0])

        dar1 = DifferencedAR1(autoreg=0.6, innovation_sd=0.3)

        with numpyro.handlers.seed(rng_seed=42):
            trajectories = dar1.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
                initial_value=initial_values,
            )

        assert trajectories.shape == (n_timepoints, n_processes)

    def test_differenced_ar1_vectorized_with_scalar_initial_value(self):
        """Test DifferencedAR1 vectorized with scalar initial value."""
        n_timepoints = 30
        n_processes = 3

        dar1 = DifferencedAR1(autoreg=0.6, innovation_sd=0.3)

        with numpyro.handlers.seed(rng_seed=42):
            trajectories = dar1.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
                initial_value=1.0,
            )

        assert trajectories.shape == (n_timepoints, n_processes)


class TestRandomWalkVectorizedSampling:
    """Test RandomWalk vectorized sampling."""

    def test_random_walk_vectorized_with_initial_values_array(self):
        """Test RandomWalk vectorized with array of initial values."""
        n_timepoints = 30
        n_processes = 4
        initial_values = jnp.array([0.0, 1.0, -1.0, 2.0])

        rw = RandomWalk(innovation_sd=0.3)

        with numpyro.handlers.seed(rng_seed=42):
            trajectories = rw.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
                initial_value=initial_values,
            )

        assert trajectories.shape == (n_timepoints, n_processes)
        # First row should be close to initial values (may vary by implementation)
        assert jnp.allclose(trajectories[0, :], initial_values)

    def test_random_walk_vectorized_with_scalar_initial_value(self):
        """Test RandomWalk vectorized with scalar initial value."""
        n_timepoints = 30
        n_processes = 3

        rw = RandomWalk(innovation_sd=0.3)

        with numpyro.handlers.seed(rng_seed=42):
            trajectories = rw.sample(
                n_timepoints=n_timepoints,
                n_processes=n_processes,
                initial_value=1.0,
            )

        assert trajectories.shape == (n_timepoints, n_processes)
        # First row should be all 1.0
        assert jnp.allclose(trajectories[0, :], 1.0)


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
        steps_small = jnp.abs(jnp.diff(trajectory_small[:, 0]))
        steps_large = jnp.abs(jnp.diff(trajectory_large[:, 0]))

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
        var_small = jnp.var(trajectory_small[burn_in:, 0])
        var_large = jnp.var(trajectory_large[burn_in:, 0])

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
        diffs_small = jnp.diff(trajectory_small[:, 0])
        diffs_large = jnp.diff(trajectory_large[:, 0])

        assert jnp.std(diffs_small) < jnp.std(diffs_large)

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
