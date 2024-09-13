"""
Test the InfectionsWithFeedback class works well within numpyro plate
"""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

import pyrenew.latent as latent
from pyrenew.deterministic import DeterministicPMF
from pyrenew.randomvariable import DistributionalVariable


def test_infections_with_feedback_plate_compatibility():
    """
    Test the InfectionsWithFeedback matching the Infections class.
    """
    I0 = jnp.array(
        np.array([0.0, 0.0, 0.0, 0.5, 0.6, 0.7, 0.8] * 5).reshape(-1, 5)
    )
    Rt = jnp.ones((10, 5))
    gen_int = jnp.array([0.4, 0.25, 0.25, 0.1])

    inf_feed_strength = DistributionalVariable(
        "inf_feed_strength", dist.Beta(1, 1)
    )
    inf_feedback_pmf = DeterministicPMF(name="inf_feedback_pmf", value=gen_int)

    # Test the InfectionsWithFeedback class
    InfectionsWithFeedback = latent.InfectionsWithFeedback(
        infection_feedback_strength=inf_feed_strength,
        infection_feedback_pmf=inf_feedback_pmf,
    )

    with numpyro.handlers.seed(rng_seed=0):
        with numpyro.plate("test_plate", 5):
            samp = InfectionsWithFeedback(
                gen_int=gen_int,
                Rt=Rt,
                I0=I0,
            )

    assert samp.rt.shape == Rt.shape
