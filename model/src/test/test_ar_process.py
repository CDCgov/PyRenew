import jax.numpy as jnp
import numpyro
from pyrenew.process import ARProcess


def test_ar_can_be_sampled():
    """
    Check that an AR process
    can be initialized and sampled from
    """
    ar1 = ARProcess(5, jnp.array([0.95]), jnp.array([0.5]))
    with numpyro.handlers.seed(rng_seed=62):
        ## can sample with and without inits
        ar1.sample(3532, inits=jnp.array([50.0]))
        ar1.sample(5023)

    ar3 = ARProcess(5, jnp.array([0.05, 0.025, 0.025]), jnp.array([0.5]))
    with numpyro.handlers.seed(rng_seed=62):
        ar3.sample(1230)
        ar3.sample(52, inits=jnp.array([50.0, 49.9, 48.2]))
