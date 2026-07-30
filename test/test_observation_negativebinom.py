# numpydoc ignore=GL08

import numpy as np
import numpy.testing as testing
import jax.numpy as jnp
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

    rates = np.array([3, 1, 4, 2, 3, 1, 4, 2, 3, 1])
    with numpyro.handlers.seed(rng_seed=223):
        sim_nb1 = negb(mu=rates, obs=rates)
        sim_nb2 = negb(mu=rates, obs=rates)

    assert isinstance(sim_nb1, ArrayLike)
    assert isinstance(sim_nb2, ArrayLike)

    testing.assert_array_equal(
        sim_nb1,
        sim_nb2,
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

    assert isinstance(sim_nb1, ArrayLike)
    assert isinstance(sim_nb2, ArrayLike)

    testing.assert_array_almost_equal(
        np.mean(sim_nb1),
        np.mean(sim_nb2),
        decimal=1,
    )

    # Sample mean should be close to the expected rate (5.0)
    testing.assert_almost_equal(np.mean(sim_nb1), 5.0, decimal=0)
    testing.assert_almost_equal(np.mean(sim_nb2), 5.0, decimal=0)


def test_negativebinom_zero_mean_has_finite_log_prob():
    """Check that zero means do not produce NaN log-probability."""
    negb = NegativeBinomialObservation(
        "negbinom_rv",
        concentration_rv=DeterministicVariable(name="concentration", value=10),
    )

    with numpyro.handlers.seed(rng_seed=223):
        tr = numpyro.handlers.trace(
            lambda: negb(mu=jnp.array([0.0, 0.0]), obs=jnp.array([0, 0]))
        ).get_trace()

    log_prob = tr["negbinom_rv"]["fn"].log_prob(tr["negbinom_rv"]["value"])
    assert jnp.all(jnp.isfinite(log_prob))
