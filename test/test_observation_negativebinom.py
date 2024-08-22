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
