# numpydoc ignore=GL08
from typing import NamedTuple

import jax.numpy as jnp
from jax.typing import ArrayLike
from pyrenew.arrayutils import PeriodicBroadcaster
from pyrenew.metaclass import (
    RandomVariable,
    SampledValue,
    _assert_sample_and_rtype,
)
from pyrenew.process.firstdifferencear import FirstDifferenceARProcess


class RtPeriodicDiffProcessSample(NamedTuple):
    """
    A container for holding the output from `process.RtPeriodicDiffProcess()`.

    Attributes
    ----------
    rt : SampledValue, optional
        The sampled Rt.
    """

    rt: SampledValue | None = None

    def __repr__(self):
        return f"RtPeriodicDiffProcessSample(rt={self.rt})"


class RtPeriodicDiffProcess(RandomVariable):
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
        offset: int,
        period_size: int,
        log_rt_prior: RandomVariable,
        autoreg: RandomVariable,
        periodic_diff_sd: RandomVariable,
        site_name: str = "rt_periodic_diff",
    ) -> None:
        """
        Default constructor for RtPeriodicDiffProcess class.

        Parameters
        ----------
        offset : int
            Relative point at which data starts, must be between 0 and
            period_size - 1.
        log_rt_prior : RandomVariable
            Log Rt prior for the first two observations.
        autoreg : RandomVariable
            Autoregressive parameter.
        periodic_diff_sd : RandomVariable
            Standard deviation of the noise.
        site_name : str, optional
            Name of the site. Defaults to "rt_periodic_diff".

        Returns
        -------
        None
        """

        self.broadcaster = PeriodicBroadcaster(
            offset=offset,
            period_size=period_size,
            broadcast_type="repeat",
        )

        self.validate(
            log_rt_prior=log_rt_prior,
            autoreg=autoreg,
            periodic_diff_sd=periodic_diff_sd,
        )

        self.period_size = period_size
        self.offset = offset
        self.log_rt_prior = log_rt_prior
        self.autoreg = autoreg
        self.periodic_diff_sd = periodic_diff_sd
        self.site_name = site_name

        return None

    @staticmethod
    def validate(
        log_rt_prior: any,
        autoreg: any,
        periodic_diff_sd: any,
    ) -> None:
        """
        Validate the input parameters.

        Parameters
        ----------
        log_rt_prior : any
            Log Rt prior for the first two observations.
        autoreg : any
            Autoregressive parameter.
        periodic_diff_sd : any
            Standard deviation of the noise.

        Returns
        -------
        None
        """

        _assert_sample_and_rtype(log_rt_prior)
        _assert_sample_and_rtype(autoreg)
        _assert_sample_and_rtype(periodic_diff_sd)

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
        duration: int,
        **kwargs,
    ) -> RtPeriodicDiffProcessSample:
        """
        Samples the periodic Rt with autoregressive difference.

        Parameters
        ----------
        duration : int
            Duration of the sequence.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, should there be any.

        Returns
        -------
        RtPeriodicDiffProcessSample
            Named tuple with "rt".
        """

        # Initial sample
        log_rt_prior = self.log_rt_prior.sample(**kwargs)[0].value
        b = self.autoreg.sample(**kwargs)[0].value
        s_r = self.periodic_diff_sd.sample(**kwargs)[0].value

        # How many periods to sample?
        n_periods = int(jnp.ceil(duration / self.period_size))

        # Running the process
        ar_diff = FirstDifferenceARProcess(autoreg=b, noise_sd=s_r)
        log_rt = ar_diff.sample(
            duration=n_periods,
            init_val=log_rt_prior[1],
            init_rate_of_change=log_rt_prior[1] - log_rt_prior[0],
        )[0]

        return RtPeriodicDiffProcessSample(
            rt=SampledValue(
                self.broadcaster(jnp.exp(log_rt.value.flatten()), duration)
            ),
        )


class RtWeeklyDiffProcess(RtPeriodicDiffProcess):
    """
    Weekly Rt with autoregressive difference.
    """

    def __init__(
        self,
        offset: int,
        log_rt_prior: RandomVariable,
        autoreg: RandomVariable,
        periodic_diff_sd: RandomVariable,
        site_name: str = "rt_weekly_diff",
    ) -> None:
        """
        Default constructor for RtWeeklyDiffProcess class.

        Parameters
        ----------
        offset : int
            Relative point at which data starts, must be between 0 and 6.
        log_rt_prior : RandomVariable
            Log Rt prior for the first two observations.
        autoreg : RandomVariable
            Autoregressive parameter.
        periodic_diff_sd : RandomVariable
            Standard deviation of the noise.
        site_name : str, optional
            Name of the site. Defaults to "rt_weekly_diff".

        Returns
        -------
        None
        """

        super().__init__(
            offset=offset,
            period_size=7,
            log_rt_prior=log_rt_prior,
            autoreg=autoreg,
            periodic_diff_sd=periodic_diff_sd,
            site_name=site_name,
        )

        return None
