"""
Test the rtweeklydiff module
"""

import jax.numpy as jnp
import numpy as np
import numpyro as npro
from jax import lax
from jax.typing import ArrayLike
from numpy.testing import assert_array_equal
from pyrenew.deterministic import DeterministicVariable
from pyrenew.process import RtWeeklyDiff


def _manual_rt_weekly_diff(
    log_seed: ArrayLike, sd: ArrayLike, b: ArrayLike
) -> ArrayLike:
    """
    'Manually' (for-loop) reconstruct the scanner process

    Parameters
    ----------
    log_seed : ArrayLike
        The initial log seed.
    sd : ArrayLike
        The noise.
    b : ArrayLike
        The autoregressive parameter.

    Returns
    -------
    ArrayLike
        The reconstructed RtWeeklyDiff process.
    """

    log_ans = np.zeros(sd.size + 2)
    log_ans[0] = log_seed[0]
    log_ans[1] = log_seed[1]
    for i in range(2, sd.size + 2):
        log_ans[i] = (
            log_ans[i - 1]
            + b[0] * (log_ans[i - 1] - log_ans[i - 2])
            + sd[i - 2]
        )

    return log_ans[2:]


def test_rtweeklydiff() -> None:
    """Checks basic functionality of the process"""

    params = {
        "n_obs": 30,
        "data_starts": 0,
        "log_rt_prior": DeterministicVariable(jnp.array([0.1, 0.2])),
        "autoreg": DeterministicVariable(jnp.array([0.7])),
        "sigma_r": DeterministicVariable(jnp.array([0.1])),
        "site_name": "test",
    }

    rtwd = RtWeeklyDiff(**params)

    assert rtwd.n_periods == 5

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
    params["data_starts"] = 5
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
    """Checks step size averages close to 0"""

    params = {
        "n_obs": 1000,
        "data_starts": 0,
        "log_rt_prior": DeterministicVariable(jnp.array([0.0, 0.0])),
        # No autoregression!
        "autoreg": DeterministicVariable(jnp.array([0.0])),
        "sigma_r": DeterministicVariable(jnp.array([0.1])),
        "site_name": "test",
    }

    rtwd = RtWeeklyDiff(**params)

    assert rtwd.n_periods == int(jnp.ceil(params["n_obs"] / 7))

    np.random.seed(223)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        rt = rtwd.sample().rt

    # Checking that the shape of the sampled Rt is correct
    assert rt.shape == (rtwd.n_obs,)

    # Checking that the sampled Rt is constant every 7 days
    for i in range(0, rtwd.n_obs, 7):
        j = jnp.min(jnp.array([i + 7, rtwd.n_obs]))

        assert_array_equal(rt[i:j], jnp.repeat(rt[i], rt[i:j].size))

    # Checking that the difference is close to zero
    assert jnp.abs(jnp.mean(rt[1:] - rt[:-1])) < 0.01

    return None


def test_rtweeklydiff_manual_reconstruction() -> None:
    """Checks that the 'manual' reconstruction is correct"""

    params = {
        "n_obs": 30,
        "data_starts": 0,
        "log_rt_prior": DeterministicVariable(jnp.array([0.1, 0.2])),
        "autoreg": DeterministicVariable(jnp.array([0.7])),
        "sigma_r": DeterministicVariable(jnp.array([0.1])),
        "site_name": "test",
    }

    rtwd = RtWeeklyDiff(**params)
    b = jnp.array([0.7])

    noise = np.random.normal(0, 0.1, 10)

    _, ans0 = lax.scan(
        f=rtwd.autoreg_process,
        init=np.hstack([params["log_rt_prior"].sample()[0], b]),
        xs=noise,
    )

    ans1 = _manual_rt_weekly_diff(
        log_seed=params["log_rt_prior"].sample()[0], sd=noise, b=b
    )

    assert_array_equal(ans0, ans1)

    return None
