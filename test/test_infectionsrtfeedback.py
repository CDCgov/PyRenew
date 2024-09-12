"""
Test the InfectionsWithFeedback class
"""

import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from jax.typing import ArrayLike
from numpy.testing import assert_array_almost_equal, assert_array_equal

import pyrenew.latent as latent
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

    T = len(Rt)
    Rt = np.array(Rt).reshape(
        T, -1
    )  # coerce from jax to use numpy-like operations
    len_gen = len(gen_int)
    infs = np.concatenate([I0.reshape(T, -1), np.zeros(Rt.shape)])
    Rt_adj = np.zeros(Rt.shape)
    inf_feedback_strength = np.array(inf_feedback_strength).reshape(T, -1)

    def compute_Rt_adj(
        Rt, inf_feedback_strength, infs, inf_feedback_pmf, len_gen, t, n
    ):  # numpydoc ignore=GL08
        return Rt[t, n] * np.exp(
            inf_feedback_strength[t, n]
            * np.dot(infs[t : t + len_gen, n], np.flip(inf_feedback_pmf))
        )

    Rt_adj = np.array(
        [
            [
                compute_Rt_adj(
                    Rt,
                    inf_feedback_strength,
                    infs,
                    inf_feedback_pmf,
                    len_gen,
                    t,
                    n,
                )
                for n in range(Rt.shape[1])
            ]
            for t in range(Rt.shape[0])
        ]
    )

    def compute_infections(
        Rt_adj, infs, len_gen, gen_int, t, n
    ):  # numpydoc ignore=GL08
        return Rt_adj[t, n] * np.dot(
            infs[t : t + len_gen, n], np.flip(gen_int)
        )

    infs[len_gen : T + len_gen] = np.array(
        [
            [
                compute_infections(Rt_adj, infs, len_gen, gen_int, t, n)
                for n in range(Rt.shape[1])
            ]
            for t in range(Rt.shape[0])
        ]
    )

    return {
        "post_initialization_infections": np.squeeze(infs[I0.shape[0] :]),
        "rt": np.squeeze(Rt_adj),
    }


@pytest.mark.parametrize(
    ["Rt", "I0"],
    [
        [
            jnp.array([0.5, 0.6, 0.7, 0.8, 2, 0.5, 2.25]),
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ],
        [
            jnp.array(
                np.array([0.5, 0.6, 0.7, 0.8, 2, 0.5, 2.25] * 3)
            ).reshape((7, 3)),
            jnp.array(
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] * 3)
            ).reshape((7, 3)),
        ],
    ],
)
def test_infectionsrtfeedback(Rt, I0):
    """
    Test the InfectionsWithFeedback matching the Infections class.
    """
    gen_int = jnp.array([0.4, 0.25, 0.25, 0.1, 0.0, 0.0, 0.0])

    # By doing the infection feedback strength 0, Rt = Rt_adjusted
    # So infection should be equal in both
    inf_feed_strength = DeterministicVariable(
        name="inf_feed_strength", value=jnp.zeros_like(Rt)
    )
    inf_feedback_pmf = DeterministicPMF(name="inf_feedback_pmf", value=gen_int)

    # Test the InfectionsWithFeedback class
    InfectionsWithFeedback = latent.InfectionsWithFeedback(
        infection_feedback_strength=inf_feed_strength,
        infection_feedback_pmf=inf_feedback_pmf,
    )

    infections = latent.Infections()

    with numpyro.handlers.seed(rng_seed=0):
        samp1 = InfectionsWithFeedback(
            gen_int=gen_int,
            Rt=Rt,
            I0=I0,
        )

        samp2 = infections(
            gen_int=gen_int,
            Rt=Rt,
            I0=I0,
        )

    assert_array_equal(
        samp1.post_initialization_infections,
        samp2.post_initialization_infections,
    )
    assert_array_equal(samp1.rt, Rt)

    return None


@pytest.mark.parametrize(
    ["Rt", "I0"],
    [
        [
            jnp.array([0.5, 0.6, 0.7, 0.8, 2, 0.5, 2.25]),
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ],
        [
            jnp.array(
                np.array([0.5, 0.6, 0.7, 0.8, 2, 0.5, 2.25] * 3)
            ).reshape((7, 3)),
            jnp.array(
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] * 3)
            ).reshape((7, 3)),
        ],
    ],
)
def test_infectionsrtfeedback_feedback(Rt, I0):
    """
    Test the InfectionsWithFeedback with feedback
    """
    gen_int = jnp.array([0.4, 0.25, 0.25, 0.1, 0.0, 0.0, 0.0])

    inf_feed_strength = DeterministicVariable(
        name="inf_feed_strength", value=0.5 * jnp.ones_like(Rt)
    )
    inf_feedback_pmf = DeterministicPMF(name="inf_feedback_pmf", value=gen_int)

    # Test the InfectionsWithFeedback class
    InfectionsWithFeedback = latent.InfectionsWithFeedback(
        infection_feedback_strength=inf_feed_strength,
        infection_feedback_pmf=inf_feedback_pmf,
    )

    infections = latent.Infections()

    with numpyro.handlers.seed(rng_seed=0):
        samp1 = InfectionsWithFeedback(
            gen_int=gen_int,
            Rt=Rt,
            I0=I0,
        )

        samp2 = infections(
            gen_int=gen_int,
            Rt=Rt,
            I0=I0,
        )

    res = _infection_w_feedback_alt(
        gen_int=gen_int,
        Rt=Rt,
        I0=I0,
        inf_feedback_strength=inf_feed_strength(),
        inf_feedback_pmf=inf_feedback_pmf(),
    )

    assert not jnp.array_equal(
        samp1.post_initialization_infections,
        samp2,
    )
    assert_array_almost_equal(
        samp1.post_initialization_infections,
        res["post_initialization_infections"],
    )
    assert_array_almost_equal(samp1.rt, res["rt"])

    return None
