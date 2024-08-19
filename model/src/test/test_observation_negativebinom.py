# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import numpy as np
import numpy.testing as testing
import numpyro
from jax.typing import ArrayLike
from pyrenew.deterministic import DeterministicVariable
from pyrenew.observation import NegativeBinomialObservation


def test_negativebinom_deterministic_obs():
    """
    Check that a deterministic NegativeBinomialObservation can sample
    """

    negb = NegativeBinomialObservation(
        "negbinom_rv",
        concentration_rv=DeterministicVariable(name="concentration", value=10),
    )

    rates = np.random.randint(1, 5, size=10)
    with numpyro.handlers.seed(rng_seed=223):
        sim_nb1 = negb(mu=rates, obs=rates)
        sim_nb2 = negb(mu=rates, obs=rates)

    assert isinstance(sim_nb1, tuple)
    assert isinstance(sim_nb2, tuple)
    assert isinstance(sim_nb1[0].value, ArrayLike)
    assert isinstance(sim_nb2[0].value, ArrayLike)

    testing.assert_array_equal(
        sim_nb1[0].value,
        sim_nb2[0].value,
    )


def test_negativebinom_random_obs():
    """
    Check that a random NegativeBinomialObservation can sample
    """

    negb = NegativeBinomialObservation(
        "negbinom_rv",
        concentration_rv=DeterministicVariable(name="concentration", value=10),
    )

    rates = np.repeat(5, 20000)
    with numpyro.handlers.seed(rng_seed=223):
        sim_nb1 = negb(mu=rates)
        sim_nb2 = negb(mu=rates)
    assert isinstance(sim_nb1, tuple)
    assert isinstance(sim_nb2, tuple)
    assert isinstance(sim_nb1[0].value, ArrayLike)
    assert isinstance(sim_nb2[0].value, ArrayLike)

    testing.assert_array_almost_equal(
        np.mean(sim_nb1[0].value),
        np.mean(sim_nb2[0].value),
        decimal=1,
    )


# def test_negativebinom_naive_vs_clipped():
#     """
#     Test that demonstrates the failure of the naive method (eps=0)
#     and the success of the method with clipping or small eps.
#     """


#     concentration_rv = DeterministicVariable(name="concentration", value=10)

#     negb_naive = NegativeBinomialObservation(
#         "negbinom_rv_naive",
#         concentration_rv=concentration_rv,
#         eps=0.0
#     )

#     negb_clipped = NegativeBinomialObservation(
#         "negbinom_rv_clipped",
#         concentration_rv=concentration_rv,
#         eps=jnp.finfo(float).eps
#     )

#     naive_failed = False
#     try:
#         with numpyro.handlers.seed(rng_seed=223):
#             negb_naive_sample = negb_naive(mu=0.0)
#     except Exception as e:
#         naive_failed = True
#         print(f"Naive method failed as expected: {e}")
#     assert naive_failed, "Naive method did not fail as expected."

#     clipped_failed = False
#     try:
#         with numpyro.handlers.seed(rng_seed=223):
#             negb_clipped_sample = negb_clipped(mu=0.0)
#     except Exception as e:
#         clipped_failed = True
#         print(f"Clipped method failed unexpectedly: {e}")
#     assert not clipped_failed, "Clipped method failed unexpectedly."

#     assert isinstance(negb_clipped_sample, tuple)
#     assert isinstance(negb_clipped_sample[0].value, ArrayLike)

#     mean_sample_value = jnp.mean(negb_clipped_sample[0].value)
#     assert mean_sample_value >= jnp.finfo(float).eps, "Clipped sample mean is unexpectedly low."
