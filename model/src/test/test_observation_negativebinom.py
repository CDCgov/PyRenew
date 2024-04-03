# -*- coding: utf-8 -*-

import numpy as np
import numpy.testing as testing
import numpyro as npro
from pyrenew.observation import NegativeBinomialObservation


def test_negativebinom_deterministic_obs():
    """
    Check that a deterministic NegativeBinomialObservation can sample
    """

    negb = NegativeBinomialObservation(concentration_prior=10)

    np.random.seed(223)
    rates = np.random.randint(1, 5, size=10)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        sim_pois1 = negb.sample(mean=rates, obs=rates)
        sim_pois2 = negb.sample(mean=rates, obs=rates)

    testing.assert_array_equal(
        sim_pois1,
        sim_pois2,
    )


def test_negativebinom_random_obs():
    """
    Check that a random NegativeBinomialObservation can sample
    """

    negb = NegativeBinomialObservation(concentration_prior=10)

    np.random.seed(223)
    rates = np.repeat(5, 20000)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        sim_pois1 = negb.sample(mean=rates)
        sim_pois2 = negb.sample(mean=rates)

    testing.assert_array_almost_equal(
        np.mean(sim_pois1),
        np.mean(sim_pois2),
        decimal=1,
    )
