# numpydoc ignore=GL08
from typing import NamedTuple

import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
from jax import lax
from jax.typing import ArrayLike
from pyrenew.metaclass import Model, RandomVariable, _assert_sample_and_rtype


class RtPeriodicDiffSample(NamedTuple):
    """
    A container for holding the output from the `process.RtWeeklyDiff.sample()`.

    Attributes
    ----------
    rt : ArrayLike
        The sampled Rt.
    """

    rt: ArrayLike | None = None

    def __repr__(self):
        return f"RtPeriodicDiffSample(rt={self.rt})"


class RtPeriodicDiff(Model):
    r"""
    Periodic Rt with autoregressive difference.

    Notes
    -----
    This class samples a periodic Rt with autoregressive difference. The
    mathematical formulation is given by:

    .. math::
        \log[\mathcal{R}^\mathrm{u}(t_3)] \sim \mathrm{Normal}\left(\log[\mathcal{R}^\mathrm{u}(t_2)] \
            + \beta \left(\log[\mathcal{R}^\mathrm{u}(t_2)] - \
             \log[\mathcal{R}^\mathrm{u}(t_1)]\right), \sigma_r \right)

    where :math:`\mathcal{R}^\mathrm{u}(t)` is the periodic reproduction number
    at time :math:`t`, :math:`\beta` is the autoregressive parameter, and
    :math:`\sigma_r` is the standard deviation of the noise.
    """

    def __init__(
        self,
        n_obs: int,
        data_starts: int,
        period_size: int,
        log_rt_prior: RandomVariable,
        autoreg: RandomVariable,
        sigma_r: RandomVariable,
        site_name: str = "rt_periodic_diff",
    ) -> None:
        """
        Default constructor for RtPeriodicDiff class.

        Parameters
        ----------
        n_obs : int
            Number of observations.
        data_starts : int
            Relative point at which data starts, must be between 0 and
            period_size - 1.
        log_rt_prior : RandomVariable
            Log Rt prior for the first two observations.
        autoreg : RandomVariable
            Autoregressive parameter.
        sigma_r : RandomVariable
            Standard deviation of the noise.
        site_name : str, optional
            Name of the site. Defaults to "rt_periodic_diff".

        Returns
        -------
        None
        """

        self.validate(
            n_obs,
            data_starts,
            log_rt_prior,
            autoreg,
            sigma_r,
        )

        self.n_obs = n_obs
        self.period_size = period_size
        self.data_starts = data_starts
        self.n_periods = jnp.ceil(n_obs / period_size).astype(int)
        self.log_rt_prior = log_rt_prior
        self.autoreg = autoreg
        self.sigma_r = sigma_r
        self.site_name = site_name

        return None

    @staticmethod
    def validate(
        n_obs: int,
        data_starts: int,
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
        data_starts : int
            Relative point at which data starts, must be between 0 and
            period_size - 1.
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

        # Data starts should be a positive integer
        assert 0 <= data_starts, (
            "data_starts should be a positive integer."
            + f"It is {data_starts}."
        )

        _assert_sample_and_rtype(prior)
        _assert_sample_and_rtype(autoreg)
        _assert_sample_and_rtype(sigma_r)

        return None

    @staticmethod
    def autoreg_process(
        dat: ArrayLike, sigma: float
    ) -> tuple[ArrayLike, float]:
        """
        Scan function for the autoregressive process.

        Parameters
        ----------
        dat : ArrayLike
            Data array with three elements: log_rt0, log_rt1, and b.
        sigma : float
            Standard deviation of the noise.

        Returns
        -------
        tuple
        """

        log_rt0, log_rt1, b = dat

        next_log_rt = log_rt1 + b * (log_rt1 - log_rt0) + sigma

        return jnp.hstack([log_rt1, next_log_rt, b]), next_log_rt

    def sample(
        self,
        duration: int | None = None,
        **kwargs,
    ) -> RtPeriodicDiffSample:
        """
        Samples the periodic Rt with autoregressive difference.

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
        RtPeriodicDiffSample
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
            sample_shape=(self.n_periods,),
        )

        # Running the process
        _, log_rt = lax.scan(
            f=self.autoreg_process,
            init=jnp.hstack([log_rt_prior, b]),
            xs=noise,
        )

        # Expanding according to the number of days
        if duration is None:
            duration = self.n_obs
        elif duration > self.n_obs:
            raise ValueError(
                f"Duration should be less than or equal to {self.n_obs}."
            )

        return RtPeriodicDiffSample(
            rt=jnp.repeat(jnp.exp(log_rt.flatten()), self.period_size)[
                self.data_starts : (self.data_starts + duration)
            ],
        )


class RtWeeklyDiff(RtPeriodicDiff):
    """
    Weekly Rt with autoregressive difference.
    """

    def __init__(
        self,
        n_obs: int,
        data_starts: int,
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
        data_starts : int
            Relative point at which data starts, must be between 0 and 6.
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

        assert 0 <= data_starts <= 6, (
            "data_starts should be between 0 and 6." + f"It is {data_starts}."
        )

        super().__init__(
            n_obs=n_obs,
            data_starts=data_starts,
            period_size=7,
            log_rt_prior=log_rt_prior,
            autoreg=autoreg,
            sigma_r=sigma_r,
            site_name=site_name,
        )

        return None
