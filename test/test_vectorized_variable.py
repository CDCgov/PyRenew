"""
Tests for pyrenew.randomvariable.vectorizedvariable
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from pyrenew.randomvariable import DistributionalVariable, VectorizedVariable


class TestVectorizedVariable:
    """Test VectorizedVariable wrapper class."""

    def test_init_and_sample(self):
        """Test VectorizedVariable initialization and sampling."""
        rv = DistributionalVariable("test", dist.Normal(0, 1.0))
        vectorized = VectorizedVariable(name="test_vectorized", rv=rv)

        with numpyro.handlers.seed(rng_seed=42):
            samples = vectorized.sample(n_groups=5)

        assert samples.shape == (5,)
        # Verify samples are actually different (not degenerate)
        assert jnp.std(samples) > 0
