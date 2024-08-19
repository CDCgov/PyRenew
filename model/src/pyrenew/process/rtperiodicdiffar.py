# numpydoc ignore=GL08
from typing import NamedTuple

import jax.numpy as jnp
import pyrenew.arrayutils as au
from pyrenew.metaclass import (
    RandomVariable,
    SampledValue,
    _assert_sample_and_rtype,
)
from pyrenew.process import ARProcess, DifferencedProcess


class RtPeriodicDiffARProcessSample(NamedTuple):
    """
    A container for holding the output from
    `process.RtPeriodicDiffARProcess()`.

    Attributes
    ----------
    rt : SampledValue, optional
        The sampled Rt.
    """

    rt: SampledValue | None = None

    def __repr__(self):
        return f"RtPeriodicDiffARProcessSample(rt={self.rt})"


class RtPeriodicDiffARProcess(RandomVariable):
    r"""
    Periodic Rt with autoregressive first differences

    Notes
    -----
    This class samples a periodic reproduction number R(t)
    by placing an AR(1) process
    on the first differences in log[R(t)]. Formally:

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
        Default constructor for RtPeriodicDiffARProcess class.

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

        self.validate(
            log_rt_rv=log_rt_rv,
            autoreg_rv=autoreg_rv,
            periodic_diff_sd_rv=periodic_diff_sd_rv,
        )

        self.name = name
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

    def sample(
        self,
        duration: int,
        **kwargs,
    ) -> RtPeriodicDiffARProcessSample:
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
        RtPeriodicDiffARProcessSample
            Named tuple with "rt".
        """

        # Initial sample
        log_rt_rv = self.log_rt_rv.sample(**kwargs)[0].value
        b = self.autoreg_rv.sample(**kwargs)[0].value
        s_r = self.periodic_diff_sd_rv.sample(**kwargs)[0].value

        # How many periods to sample?
        n_periods = (duration + self.period_size - 1) // self.period_size

        # Running the process
        ar_diff = DifferencedProcess(
            name="log_rt",
            fundamental_process=ARProcess(name="first_diff_log_rt_ar"),
            differencing_order=1,
        )

        log_rt = ar_diff(
            n=n_periods,
            init_vals=jnp.array([log_rt_rv[1]]),
            autoreg=b,
            noise_sd=s_r,
            fundamental_process_init_vals=jnp.array(
                [log_rt_rv[1] - log_rt_rv[0]]
            ),
        )[0]

        return RtPeriodicDiffARProcessSample(
            rt=SampledValue(
                au.repeat_until_n(
                    data=jnp.exp(log_rt.value),
                    n_timepoints=duration,
                    offset=self.offset,
                    period_size=self.period_size,
                ),
                t_start=self.t_start,
                t_unit=self.t_unit,
            ),
        )


class RtWeeklyDiffARProcess(RtPeriodicDiffARProcess):
    """
    Weekly Rt with autoregressive first differences.
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
        Default constructor for RtWeeklyDiffARProcess class.

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
