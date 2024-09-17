"""
Test functions from the latent.infection_functions
submodule
"""

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from pyrenew.latent import infection_functions as inf


def test_compute_infections_from_rt_with_feedback():
    """
    test that the implementation of infection
    feedback is as expected
    """

    # if feedback is zero, results should be
    # equivalent to compute_infections_from_rt
    # and Rt_adjusted should be Rt_raw

    gen_ints = [
        jnp.array([0.25, 0.5, 0.25]),
        jnp.array([1.0]),
        jnp.ones(35) / 35,
    ]

    inf_pmfs = [jnp.ones_like(x) for x in gen_ints]

    I0s = [
        jnp.array([0.235, 6.523, 100052.0]),
        jnp.array([5.0]),
        3.5235 * jnp.ones(35),
    ]

    Rts_raw = [
        jnp.array([1.25, 0.52, 23.0, 1.0]),
        jnp.ones(500),
        jnp.zeros(253),
    ]

    for I0, gen_int, inf_pmf in zip(I0s, gen_ints, inf_pmfs):
        for Rt_raw in Rts_raw:
            (
                infs_feedback,
                Rt_adj,
            ) = inf.compute_infections_from_rt_with_feedback(
                I0, Rt_raw, jnp.zeros_like(Rt_raw), gen_int, inf_pmf
            )

            assert_array_equal(
                inf.compute_infections_from_rt(I0, Rt_raw, gen_int),
                infs_feedback,
            )

            assert_array_equal(Rt_adj, Rt_raw)
    return None


@pytest.mark.parametrize(
    ["I0", "gen_int", "inf_pmf", "Rt_raw"],
    [
        [
            jnp.array([[5.0, 0.2]]),
            jnp.array([1.0]),
            jnp.array([1.0]),
            jnp.ones((5, 2)),
        ],
        [
            3.5235 * jnp.ones((35, 3)),
            jnp.ones(35) / 35,
            jnp.ones(35),
            jnp.zeros((253, 3)),
        ],
    ],
)
def test_compute_infections_from_rt_with_feedback_2d(
    I0, gen_int, inf_pmf, Rt_raw
):
    """
    Test implementation of infection feedback
    when I0 and Rt are 2d arrays.
    """
    (
        infs_feedback,
        Rt_adj,
    ) = inf.compute_infections_from_rt_with_feedback(
        I0, Rt_raw, jnp.zeros_like(Rt_raw), gen_int, inf_pmf
    )

    assert_array_equal(
        inf.compute_infections_from_rt(I0, Rt_raw, gen_int),
        infs_feedback,
    )

    assert_array_equal(Rt_adj, Rt_raw)

    return None
