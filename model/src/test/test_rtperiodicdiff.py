"""
Test the rtweeklydiff module
"""

import jax.numpy as jnp
import numpy as np
import numpyro
from jax import lax
from jax.typing import ArrayLike
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pyrenew.deterministic import DeterministicVariable
from pyrenew.process import RtWeeklyDiffProcess


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
        The reconstructed RtWeeklyDiffProcess process.
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
        "name": "test",
        "offset": 0,
        "log_rt_rv": DeterministicVariable(
            # name="log_rt", value=jnp.array([0.1, 0.2])
            name="log_rt",
            value=jnp.array([4.774, 4.803]),
        ),
        "autoreg_rv": DeterministicVariable(
            # name="autoreg_rv", value=jnp.array([0.7])
            name="autoreg_rv",
            value=jnp.array([0.041]),
        ),
        "periodic_diff_sd_rv": DeterministicVariable(
            # name="periodic_diff_sd_rv", value=jnp.array([0.1])
            name="periodic_diff_sd_rv",
            value=jnp.array([0.009]),
        ),
    }
    duration = 30

    rtwd = RtWeeklyDiffProcess(**params)

    with numpyro.handlers.seed(rng_seed=223):
        rt = rtwd(duration=duration).rt.value

    # Checking that the shape of the sampled Rt is correct
    assert rt.shape == (duration,)

    # Checking that the sampled Rt is constant every 7 days
    for i in range(0, 28, 7):
        assert_array_equal(rt[i : i + 7], jnp.repeat(rt[i], 7))
    assert_array_equal(rt[28:duration], jnp.repeat(rt[28], 2))

    # Checking start off a different day of the week
    params["offset"] = 5
    rtwd = RtWeeklyDiffProcess(**params)

    with numpyro.handlers.seed(rng_seed=223):
        rt2 = rtwd(duration=duration).rt.value

    # Checking that the shape of the sampled Rt is correct
    assert rt2.shape == (duration,)

    # This time series should be the same as the previous one,
    # but shifted by 5 days
    assert_array_equal(rt[5:], rt2[:-5])

    return None


def test_rtweeklydiff_no_autoregressive() -> None:
    """Checks step size averages close to 0"""

    params = {
        "name": "test",
        "offset": 0,
        "log_rt_rv": DeterministicVariable(
            name="log_rt", value=jnp.array([0.0, 0.0])
        ),
        # No autoregression!
        "autoreg_rv": DeterministicVariable(
            name="autoreg_rv", value=jnp.array([0.0])
        ),
        "periodic_diff_sd_rv": DeterministicVariable(
            name="periodic_diff_sd_rv",
            value=jnp.array([0.1]),
        ),
    }

    rtwd = RtWeeklyDiffProcess(**params)

    duration = 1000

    with numpyro.handlers.seed(rng_seed=323):
        rt = rtwd(duration=duration).rt.value

    # Checking that the shape of the sampled Rt is correct
    assert rt.shape == (duration,)

    # Checking that the sampled Rt is constant every 7 days
    for i in range(0, duration, 7):
        j = jnp.min(jnp.array([i + 7, duration]))

        assert_array_equal(rt[i:j], jnp.repeat(rt[i], rt[i:j].size))

    # Checking that the difference is close to zero
    assert jnp.abs(jnp.mean(rt[1:] - rt[:-1])) < 0.01

    return None


def test_rtweeklydiff_manual_reconstruction() -> None:
    """Checks that the 'manual' reconstruction is correct"""

    params = {
        "name": "test",
        "offset": 0,
        "log_rt_rv": DeterministicVariable(
            name="log_rt",
            value=jnp.array([0.1, 0.2]),
        ),
        "autoreg_rv": DeterministicVariable(
            name="autoreg_rv", value=jnp.array([0.7])
        ),
        "periodic_diff_sd_rv": DeterministicVariable(
            name="periodic_diff_sd_rv",
            value=jnp.array([0.1]),
        ),
    }

    rtwd = RtWeeklyDiffProcess(**params)
    b = jnp.array([0.7])

    noise = np.random.normal(0, 0.1, 10)

    _, ans0 = lax.scan(
        f=rtwd.autoreg_process,
        init=np.hstack([params["log_rt_rv"]()[0].value, b]),
        xs=noise,
    )

    ans1 = _manual_rt_weekly_diff(
        log_seed=params["log_rt_rv"]()[0].value, sd=noise, b=b
    )

    assert_array_almost_equal(ans0, ans1)

    return None


def test_rtperiodicdiff_smallsample():
    """Checks basic functionality of the process with a small sample size."""

    params = {
        "name": "test",
        "offset": 0,
        "log_rt_rv": DeterministicVariable(
            name="log_rt",
            value=jnp.array([0.1, 0.2]),
        ),
        "autoreg_rv": DeterministicVariable(
            name="autoreg_rv", value=jnp.array([0.7])
        ),
        "periodic_diff_sd_rv": DeterministicVariable(
            name="periodic_diff_sd_rv",
            value=jnp.array([0.1]),
        ),
    }

    rtwd = RtWeeklyDiffProcess(**params)

    with numpyro.handlers.seed(rng_seed=223):
        rt = rtwd(duration=6).rt.value

    # Checking that the shape of the sampled Rt is correct
    assert rt.shape == (6,)

    # Check that all values in rt are the same
    assert jnp.all(rt == rt[0])
