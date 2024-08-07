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
        name: str,
        offset: int,
        period_size: int,
        log_rt_rv: RandomVariable,
        autoreg_rv: RandomVariable,
        periodic_diff_sd_rv: RandomVariable,
    ) -> None:
        """
        Default constructor for RtPeriodicDiffProcess class.

        Parameters
        ----------
        name : str
            Name of the site.
        offset : int
            Relative point at which data starts, must be between 0 and
            period_size - 1.
        log_rt_rv : RandomVariable
            Log Rt prior for the first two observations.
        autoreg_rv : RandomVariable
            Autoregressive parameter.
        periodic_diff_sd_rv : RandomVariable
            Standard deviation of the noise.

        Returns
        -------
        None
        """
        self.name = name
        self.broadcaster = PeriodicBroadcaster(
            offset=offset,
            period_size=period_size,
            broadcast_type="repeat",
        )

        self.validate(
            log_rt_rv=log_rt_rv,
            autoreg_rv=autoreg_rv,
            periodic_diff_sd_rv=periodic_diff_sd_rv,
        )

        self.period_size = period_size
        self.offset = offset
        self.log_rt_rv = log_rt_rv
        self.autoreg_rv = autoreg_rv
        self.periodic_diff_sd_rv = periodic_diff_sd_rv

        return None

    @staticmethod
    def validate(
        log_rt_rv: any,
        autoreg_rv: any,
        periodic_diff_sd_rv: any,
    ) -> None:
        """
        Validate the input parameters.

        Parameters
        ----------
        log_rt_rv : any
            Log Rt prior for the first two observations.
        autoreg_rv : any
            Autoregressive parameter.
        periodic_diff_sd_rv : any
            Standard deviation of the noise.

        Returns
        -------
        None
        """

        _assert_sample_and_rtype(log_rt_rv)
        _assert_sample_and_rtype(autoreg_rv)
        _assert_sample_and_rtype(periodic_diff_sd_rv)

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
        log_rt_rv = self.log_rt_rv.sample(**kwargs)[0].value
        b = self.autoreg_rv.sample(**kwargs)[0].value
        s_r = self.periodic_diff_sd_rv.sample(**kwargs)[0].value

        # How many periods to sample?
        n_periods = (duration + self.period_size - 1) // self.period_size

        # Running the process
        ar_diff = FirstDifferenceARProcess(self.name, autoreg=b, noise_sd=s_r)
        log_rt = ar_diff.sample(
            duration=n_periods,
            init_val=log_rt_rv[1],
            init_rate_of_change=log_rt_rv[1] - log_rt_rv[0],
        )[0]

        return RtPeriodicDiffProcessSample(
            rt=SampledValue(
                self.broadcaster(jnp.exp(log_rt.value.flatten()), duration),
                t_start=self.t_start,
                t_unit=self.t_unit,
            ),
        )


class RtWeeklyDiffProcess(RtPeriodicDiffProcess):
    """
    Weekly Rt with autoregressive difference.
    """

    def __init__(
        self,
        name: str,
        offset: int,
        log_rt_rv: RandomVariable,
        autoreg_rv: RandomVariable,
        periodic_diff_sd_rv: RandomVariable,
    ) -> None:
        """
        Default constructor for RtWeeklyDiffProcess class.

        Parameters
        ----------
        name : str
            Name of the site.
        offset : int
            Relative point at which data starts, must be between 0 and 6.
        log_rt_rv : RandomVariable
            Log Rt prior for the first two observations.
        autoreg_rv : RandomVariable
            Autoregressive parameter.
        periodic_diff_sd_rv : RandomVariable
            Standard deviation of the noise.

        Returns
        -------
        None
        """

        super().__init__(
            name=name,
            offset=offset,
            period_size=7,
            log_rt_rv=log_rt_rv,
            autoreg_rv=autoreg_rv,
            periodic_diff_sd_rv=periodic_diff_sd_rv,
        )

        return None
