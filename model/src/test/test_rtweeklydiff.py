"""
Test the rtweeklydiff module
"""

import jax.numpy as jnp
import numpy as np
import numpyro as npro
from numpy.testing import assert_array_equal
from pyrenew.deterministic import DeterministicVariable
from pyrenew.process.rtweeklydiff import RtWeeklyDiff


def test_rtweeklydiff() -> None:
    """Checks basic functionality of the process"""

    params = {
        "n_obs": 30,
        "weekday_data_starts": 0,
        "log_rt_prior": DeterministicVariable(jnp.array([0.1, 0.2])),
        "autoreg": DeterministicVariable(jnp.array([0.7])),
        "sigma_r": DeterministicVariable(jnp.array([0.1])),
        "site_name": "test",
    }

    rtwd = RtWeeklyDiff(**params)

    assert rtwd.n_weeks == 5

    np.random.seed(223)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        rt = rtwd.sample().rt

    # Checking that the shape of the sampled Rt is correct
    assert rt.shape == (30,)

    # Checking that the sampled Rt is constant every 7 days
    for i in range(0, 28, 7):
        assert_array_equal(rt[i : i + 7], jnp.repeat(rt[i], 7))
    assert_array_equal(rt[28:30], jnp.repeat(rt[28], 2))

    # Checking start off a different day of the week
    np.random.seed(223)
    params["weekday_data_starts"] = 5
    rtwd = RtWeeklyDiff(**params)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        rt2 = rtwd.sample().rt

    # Checking that the shape of the sampled Rt is correct
    assert rt2.shape == (30,)

    # This time series should be the same as the previous one, but shifted by
    # 5 days
    assert_array_equal(rt[5:], rt2[:-5])

    return None


def test_rtweeklydiff_no_autoregressive() -> None:
    """Checks convergence to mean when no autoregression"""

    params = {
        "n_obs": 350,
        "weekday_data_starts": 0,
        "log_rt_prior": DeterministicVariable(jnp.array([0.0, 0.0])),
        # No autoregression!
        "autoreg": DeterministicVariable(jnp.array([0.0])),
        "sigma_r": DeterministicVariable(jnp.array([0.1])),
        "site_name": "test",
    }

    rtwd = RtWeeklyDiff(**params)

    assert rtwd.n_weeks == 50

    np.random.seed(223)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        rt = rtwd.sample().rt

    # Checking that the shape of the sampled Rt is correct
    assert rt.shape == (350,)

    # Checking that the sampled Rt is constant every 7 days
    for i in range(0, 350, 7):
        assert_array_equal(rt[i : i + 7], jnp.repeat(rt[i], 7))

    # Checking that the mean is approx to 1
    assert jnp.abs(jnp.mean(rt) - 1.0) < 0.01

    return None


# test_rtweeklydiff()
# test_rtweeklydiff_no_autoregressive()
