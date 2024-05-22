from typing import NamedTuple

import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
from jax import lax
from jax.typing import ArrayLike
from pyrenew.metaclass import Model, RandomVariable, _assert_sample_and_rtype


class RtWeeklyDiffSample(NamedTuple):
    """
    A container for holding the output from the `process.RtWeeklyDiff.sample()`.

    Attributes
    ----------
    rt : ArrayLike
        The sampled Rt.
    """

    rt: ArrayLike | None = None

    def __repr__(self):
        return f"RtWeeklyDiffSample(rt={self.rt})"


class RtWeeklyDiff(Model):
    r"""
    Weekly Rt with autoregressive difference.

    Notes
    -----
    This class samples a weekly Rt with autoregressive difference. The
    mathematical formulation is given by:

    .. math::
        \log[\mathcal{R}^\mathrm{u}(t_3)] \sim \mathrm{Normal}\left(\log[\mathcal{R}^\mathrm{u}(t_2)] \
            + \beta \left(\log[\mathcal{R}^\mathrm{u}(t_2)] - \
             \log[\mathcal{R}^\mathrm{u}(t_1)]\right), \sigma_r \right)

    where :math:`\mathcal{R}^\mathrm{u}(t)` is the weekly reproduction number
    at week :math:`t`, :math:`\beta` is the autoregressive parameter, and
    :math:`\sigma_r` is the standard deviation of the noise.
    """

    def __init__(
        self,
        n_obs: int,
        weekday_data_starts: int,
        log_rt_prior: RandomVariable,
        autoreg: RandomVariable,
        sigma_r: RandomVariable,
        site_name: str = "rt_weekly_diff",
    ) -> None:
        """
        Default constructor for RtWeeklyDiff class.

        Parameters
        ----------
        n_obs : int
            Number of observations.
        weekday_data_starts : int
            Weekday data starts, must be between 0 and 6, 0 beign Sunday.
        log_rt_prior : RandomVariable
            Log Rt prior for the first two observations.
        autoreg : RandomVariable
            Autoregressive parameter.
        sigma_r : RandomVariable
            Standard deviation of the noise.
        site_name : str, optional
            Name of the site. Defaults to "rt_weekly_diff".

        Returns
        -------
        None
        """

        self.validate(
            n_obs,
            weekday_data_starts,
            log_rt_prior,
            autoreg,
            sigma_r,
        )

        self.n_obs = n_obs
        self.weekday_data_starts = weekday_data_starts
        self.n_weeks = jnp.ceil(n_obs / 7).astype(int)
        self.log_rt_prior = log_rt_prior
        self.autoreg = autoreg
        self.sigma_r = sigma_r
        self.site_name = site_name

        return None

    @staticmethod
    def validate(
        n_obs: int,
        weekday_data_starts: int,
        prior: any,
        autoreg: any,
        sigma_r: any,
    ) -> None:
        """
        Validate the input parameters.

        Parameters
        ----------
        n_obs : int
            Number of observations.
        weekday_data_starts : int
            Weekday data starts.
        prior : any
            Log Rt prior for the first two observations.
        autoreg : any
            Autoregressive parameter.
        sigma_r : any
            Standard deviation of the noise.

        Returns
        -------
        None
        """

        # Nweeks should be a positive integer
        assert n_obs > 0, f"n_obs should be a positive integer. It is {n_obs}."

        # Weekday data starts should be a positive integer between 0 and 6
        assert 0 <= weekday_data_starts <= 6, (
            "weekday_data_starts should be a positive integer between 0 and 6."
            + f"It is {weekday_data_starts}."
        )

        _assert_sample_and_rtype(prior)
        _assert_sample_and_rtype(autoreg)
        _assert_sample_and_rtype(sigma_r)

        return None

    def sample(
        self,
        duration: int | None = None,
        **kwargs,
    ) -> RtWeeklyDiffSample:
        """
        Samples the weekly Rt with autoregressive difference.

        Parameters
        ----------
        duration : int, optional
            Duration of the sequence. Defaults to None, in which case it is
            set to the number of observations (`self.n_obs`).
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, should there be any.

        Returns
        -------
        RtWeeklyDiffSample
            Named tuple with "rt".
        """

        # Initial sample
        log_rt_prior = self.log_rt_prior.sample(**kwargs)[0]
        b = self.autoreg.sample(**kwargs)[0]
        s_r = self.sigma_r.sample(**kwargs)[0]

        # Sample noise
        noise = npro.sample(
            self.site_name + "_error",
            dist.Normal(0, s_r),
            sample_shape=(self.n_weeks,),
        )

        # Building the scanner
        def _rt_scanner(log_rts, sigma):
            next_log_rt = log_rts[1] + b * (log_rts[1] - log_rts[0]) + sigma
            return jnp.hstack([log_rts[1:], next_log_rt]), next_log_rt

        # Scanning
        _, log_rt = lax.scan(
            f=_rt_scanner,
            init=log_rt_prior,
            xs=noise,
        )

        # Expanding according to the number of days
        if duration is None:
            duration = self.n_obs
        elif duration > self.n_obs:
            raise ValueError(
                f"Duration should be less than or equal to {self.n_obs}."
            )

        return RtWeeklyDiffSample(
            rt=jnp.repeat(jnp.exp(log_rt.flatten()), 7)[
                self.weekday_data_starts : (
                    self.weekday_data_starts + duration
                )
            ],
        )
