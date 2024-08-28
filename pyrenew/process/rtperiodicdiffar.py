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
        log_rt_init_rv: RandomVariable,
        autoreg_rv: RandomVariable,
        periodic_diff_sd_rv: RandomVariable,
        ar_process_suffix: str = "_first_diff_ar_process_noise",
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
        log_rt_init_rv : RandomVariable
            Log Rt prior for the first two observations.
        autoreg_rv : RandomVariable
            Autoregressive parameter.
        periodic_diff_sd_rv : RandomVariable
            Standard deviation of the noise.
        ar_process_suffix : str
            Suffix to append to the :class:`RandomVariable`'s ``name``
            when naming the :class:`RandomVariable` that represents
            the underlying AR process noise.
            Default "_first_diff_ar_process_noise".

        Returns
        -------
        None
        """

        self.validate(
            log_rt_init_rv=log_rt_init_rv,
            autoreg_rv=autoreg_rv,
            periodic_diff_sd_rv=periodic_diff_sd_rv,
        )

        self.name = name
        self.period_size = period_size
        self.offset = offset
        self.log_rt_init_rv = log_rt_init_rv
        self.autoreg_rv = autoreg_rv
        self.periodic_diff_sd_rv = periodic_diff_sd_rv
        self.ar_diff = DifferencedProcess(
            fundamental_process=ARProcess(
                noise_rv_name=f"{name}{ar_process_suffix}"
            ),
            differencing_order=1,
        )

        return None

    @staticmethod
    def validate(
        log_rt_init_rv: any,
        autoreg_rv: any,
        periodic_diff_sd_rv: any,
    ) -> None:
        """
        Validate the input parameters.

        Parameters
        ----------
        log_rt_init_rv : any
            Log Rt prior for the first two observations.
        autoreg_rv : any
            Autoregressive parameter.
        periodic_diff_sd_rv : any
            Standard deviation of the noise.

        Returns
        -------
        None
        """

        _assert_sample_and_rtype(log_rt_init_rv)
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
        log_rt_init = self.log_rt_init_rv.sample(**kwargs)[0].value
        autoreg = self.autoreg_rv.sample(**kwargs)[0].value
        noise_sd = self.periodic_diff_sd_rv.sample(**kwargs)[0].value

        # How many periods to sample?
        n_periods = (duration + self.period_size - 1) // self.period_size

        # Running the process

        log_rt = self.ar_diff(
            n=n_periods,
            init_vals=jnp.array([log_rt_init[0]]),
            autoreg=autoreg,
            noise_sd=noise_sd,
            fundamental_process_init_vals=jnp.array(
                [log_rt_init[1] - log_rt_init[0]]
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
