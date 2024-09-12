"""
Test that key :class:`RandomVariable`
classes behave as expected in a
:func:`numpyro.plate` context.
"""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest

from pyrenew.process import ARProcess
from pyrenew.randomvariable import DistributionalVariable


@pytest.mark.parametrize(
    ["random_variable", "constructor_args", "sample_args"],
    [
        [
            ARProcess,
            dict(),
            dict(
                noise_name="ar_noise",
                n=100,
                autoreg=jnp.array([0.25, 0.1]),
                init_vals=jnp.array([15.0, 50.2]),
                noise_sd=jnp.array([0.5, 1.5]),
            ),
        ]
    ],
)
def test_single_plate_sampling(random_variable, constructor_args, sample_args):
    """
    Test that the output of vectorized
    scans can be sent into plate contexts
    successfully
    """
    with numpyro.handlers.seed(rng_seed=5):
        scanned_rv = random_variable(**constructor_args)
        scanned_output = scanned_rv(**sample_args)
        with numpyro.plate("test_plate", jnp.shape(scanned_output)[-1]):
            plated_rv = DistributionalVariable("test", dist.Normal(0, 1))
            plated_samp = plated_rv()
            output = scanned_output + plated_samp
    assert output.shape == scanned_output.shape
