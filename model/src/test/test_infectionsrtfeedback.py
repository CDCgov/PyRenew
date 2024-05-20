"""
Test the InfectionsWithFeedback class
"""

import jax.numpy as jnp
import numpyro as npro
import pyrenew.latent as latent
from numpy.testing import assert_array_equal
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable


def test_infectionsrtfeedback():
    """
    Test the InfectionsWithFeedback matching the Infections class.
    """

    Rt = jnp.array([0.5, 0.6, 0.7, 0.8])
    I0 = jnp.array([1.0])
    gen_int = jnp.array([0.25, 0.25, 0.25, 0.25])

    # By doing the infection feedback strength 0, Rt = Rt_adjusted
    # So infection should be equal in both
    inf_feed_strength = DeterministicVariable(jnp.array([0.0]))
    inf_feedback_pmf = DeterministicPMF(gen_int)

    # Test the InfectionsWithFeedback class
    InfectionsWithFeedback = latent.InfectionsWithFeedback(
        infection_feedback_strength=inf_feed_strength,
        infection_feedback_pmf=inf_feedback_pmf,
    )

    infections = latent.Infections()

    with npro.handlers.seed(rng_seed=0):
        samp1 = InfectionsWithFeedback.sample(
            gen_int=gen_int,
            Rt=Rt,
            I0=I0,
        )

        samp2 = infections.sample(
            gen_int=gen_int,
            Rt=Rt,
            I0=I0,
        )

    assert_array_equal(samp1.infections, samp2.infections)
    assert_array_equal(samp1.rt, Rt)

    return None


def test_infectionsrtfeedback_feedback():
    """
    Test the InfectionsWithFeedback with feedback
    """

    Rt = jnp.array([0.5, 0.6, 0.7, 0.8])
    I0 = jnp.array([1.0])
    gen_int = jnp.array([0.25, 0.25, 0.25, 0.25])

    inf_feed_strength = DeterministicVariable(jnp.array([0.5]))
    inf_feedback_pmf = DeterministicPMF(gen_int)

    # Test the InfectionsWithFeedback class
    InfectionsWithFeedback = latent.InfectionsWithFeedback(
        infection_feedback_strength=inf_feed_strength,
        infection_feedback_pmf=inf_feedback_pmf,
    )

    infections = latent.Infections()

    with npro.handlers.seed(rng_seed=0):
        samp1 = InfectionsWithFeedback.sample(
            gen_int=gen_int,
            Rt=Rt,
            I0=I0,
        )

        samp2 = infections.sample(
            gen_int=gen_int,
            Rt=Rt,
            I0=I0,
        )

    assert not jnp.array_equal(samp1.infections, samp2.infections)

    return None
