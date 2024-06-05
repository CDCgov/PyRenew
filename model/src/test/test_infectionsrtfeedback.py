"""
Test the InfectionsWithFeedback class
"""

import jax.numpy as jnp
import numpy as np
import numpyro as npro
import pyrenew.latent as latent
from jax.typing import ArrayLike
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable


def _infection_w_feedback_alt(
    gen_int: ArrayLike,
    Rt: ArrayLike,
    I0: ArrayLike,
    inf_feedback_strength: ArrayLike,
    inf_feedback_pmf: ArrayLike,
) -> tuple:
    """
    Calculate the infections with feedback.
    Parameters
    ----------
    gen_int : ArrayLike
        Generation interval.
    Rt : ArrayLike
        Reproduction number.
    I0 : ArrayLike
        Initial infections.
    inf_feedback_strength : ArrayLike
        Infection feedback strength.
    inf_feedback_pmf : ArrayLike
        Infection feedback pmf.

    Returns
    -------
    tuple
    """

    Rt = np.array(Rt)  # coerce from jax to use numpy-like operations
    T = len(Rt)
    len_gen = len(gen_int)
    I_vec = np.concatenate([I0, np.zeros(T)])
    Rt_adj = np.zeros(T)

    for t in range(T):
        Rt_adj[t] = Rt[t] * np.exp(
            inf_feedback_strength[t]
            * np.dot(I_vec[t : t + len_gen], np.flip(inf_feedback_pmf))
        )

        I_vec[t + len_gen] = Rt_adj[t] * np.dot(
            I_vec[t : t + len_gen], np.flip(gen_int)
        )

    return {"infections": I_vec, "rt": Rt_adj}


def test_infectionsrtfeedback():
    """
    Test the InfectionsWithFeedback matching the Infections class.
    """

    Rt = jnp.array([0.5, 0.6, 0.7, 0.8, 2, 0.5, 2.25])
    I0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    gen_int = jnp.array([0.4, 0.25, 0.25, 0.1, 0.0, 0.0, 0.0])

    # By doing the infection feedback strength 0, Rt = Rt_adjusted
    # So infection should be equal in both
    inf_feed_strength = DeterministicVariable(
        jnp.zeros_like(Rt), name="inf_feed_strength"
    )
    inf_feedback_pmf = DeterministicPMF(gen_int, name="inf_feedback_pmf")

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

    Rt = jnp.array([0.5, 0.6, 1.5, 2.523, 0.7, 0.8])
    I0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    gen_int = jnp.array([0.4, 0.25, 0.25, 0.1, 0.0, 0.0, 0.0])

    inf_feed_strength = DeterministicVariable(
        jnp.repeat(0.5, len(Rt)), name="inf_feed_strength"
    )
    inf_feedback_pmf = DeterministicPMF(gen_int, name="inf_feedback_pmf")

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

    res = _infection_w_feedback_alt(
        gen_int=gen_int,
        Rt=Rt,
        I0=I0,
        inf_feedback_strength=inf_feed_strength.sample()[0],
        inf_feedback_pmf=inf_feedback_pmf.sample()[0],
    )

    assert not jnp.array_equal(samp1.infections, samp2.infections)
    assert_array_almost_equal(samp1.infections, res["infections"])
    assert_array_almost_equal(samp1.rt, res["rt"])

    return None
