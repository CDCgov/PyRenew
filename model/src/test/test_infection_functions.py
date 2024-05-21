"""
Test functions from the latent.infection_functions
submodule
"""

from pyrenew.latent import infection_functions as inf
import jax.numpy as jnp
from numpy.testing import assert_array_equal


def test_sample_infections_with_feedback():
    """
    test that the implementation of infection
    feedback is as expected
    """

    # if feedback is zero, results should be
    # equivalent to sample_infections_rt
    # and Rt_adjusted should be Rt_raw

    gen_ints = [
        jnp.array([0.25,0.5, 0.25]),
        jnp.array([1.]),
        jnp.ones(35) / 35
    ]

    inf_pmfs = [
        jnp.ones_like(x) for x in gen_ints]

    I0s = [
        jnp.array([.235, 6.523, 100052.]),
        jnp.array([5.]),
        3.5235 * jnp.ones(35)
    ]

    Rts_raw = [
        jnp.array([1.25, 0.52, 23., 1.0]),
        jnp.ones(500),
        jnp.zeros(253)
    ]

    for I0, gen_int, inf_pmf in zip(
            I0s,
            gen_ints,
            inf_pmfs):
        for Rt_raw in Rts_raw:
            infs_feedback, Rt_adj = inf.sample_infections_with_feedback(
                I0,
                Rt_raw,
                jnp.zeros_like(Rt_raw),
                gen_int,
                inf_pmf)

            assert_array_equal(
                inf.sample_infections_rt(
                    I0,
                    Rt_raw,
                    gen_int
                ),
                infs_feedback)

            assert_array_equal(
                Rt_adj,
                Rt_raw)
            pass
        pass
    return None
