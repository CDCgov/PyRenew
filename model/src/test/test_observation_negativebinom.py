# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import numpy as np
import numpy.testing as testing
import numpyro as npro
from pyrenew.deterministic import DeterministicVariable
from pyrenew.observation import NegativeBinomialObservation


def test_negativebinom_deterministic_obs():
    """
    Check that a deterministic NegativeBinomialObservation can sample
    """

    negb = NegativeBinomialObservation(
        "negbinom_rv",
        concentration_rv=DeterministicVariable(10, name="concentration"),
    )

    np.random.seed(223)
    rates = np.random.randint(1, 5, size=10)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        sim_pois1, *_ = negb(mu=rates, obs=rates)
        sim_pois2, *_ = negb(mu=rates, obs=rates)

    testing.assert_array_equal(
        sim_pois1.array,
        sim_pois2.array,
    )


def test_negativebinom_random_obs():
    """
    Check that a random NegativeBinomialObservation can sample
    """

    negb = NegativeBinomialObservation(
        "negbinom_rv",
        concentration_rv=DeterministicVariable(10, "concentration"),
    )

    np.random.seed(223)
    rates = np.repeat(5, 20000)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        sim_pois1, *_ = negb(mu=rates)
        sim_pois2, *_ = negb(mu=rates)

    testing.assert_array_almost_equal(
        np.mean(sim_pois1.array),
        np.mean(sim_pois2.array),
        decimal=1,
    )
