"""
Test to verify Infection and InfectionsWithFeedback class
return error when input array shape for I0 and Rt are invalid
"""

import jax.numpy as jnp
import numpy as np
import numpyro
import pytest

import pyrenew.latent as latent
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable


def test_infections_with_feedback_invalid_inputs():
    """
    Test the InfectionsWithFeedback class cannot
    be sampled when Rt and I0 have invalid input shapes
    """
    I0_1d = jnp.array([0.5, 0.6, 0.7, 0.8])
    I0_2d = jnp.array(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] * 3)
    ).reshape((7, -1))
    Rt = jnp.ones(10)
    gen_int = jnp.array([0.4, 0.25, 0.25, 0.1, 0.0, 0.0, 0.0])

    inf_feed_strength = DeterministicVariable(
        name="inf_feed_strength", value=0.5
    )
    inf_feedback_pmf = DeterministicPMF(name="inf_feedback_pmf", value=gen_int)

    # Test the InfectionsWithFeedback class
    InfectionsWithFeedback = latent.InfectionsWithFeedback(
        infection_feedback_strength=inf_feed_strength,
        infection_feedback_pmf=inf_feedback_pmf,
    )

    infections = latent.Infections()

    with numpyro.handlers.seed(rng_seed=0):
        with pytest.raises(
            ValueError,
            match="Initial infections must be at least as long as the generation interval.",
        ):
            InfectionsWithFeedback(
                gen_int=gen_int,
                Rt=Rt,
                I0=I0_1d,
            )

        with pytest.raises(
            ValueError,
            match="Initial infections vector must be at least as long as the generation interval.",
        ):
            infections(
                gen_int=gen_int,
                Rt=Rt,
                I0=I0_1d,
            )

        with pytest.raises(
            ValueError,
            match="Initial infections and Rt must have the same batch shapes.",
        ):
            InfectionsWithFeedback(
                gen_int=gen_int,
                Rt=Rt,
                I0=I0_2d,
            )

        with pytest.raises(
            ValueError,
            match="Initial infections and Rt must have the same batch shapes.",
        ):
            infections(
                gen_int=gen_int,
                Rt=Rt,
                I0=I0_2d,
            )
