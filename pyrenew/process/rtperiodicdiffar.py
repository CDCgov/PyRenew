# numpydoc ignore=GL08

import jax.numpy as jnp
from jax.typing import ArrayLike

import pyrenew.arrayutils as au
from pyrenew.metaclass import RandomVariable, _assert_sample_and_rtype
from pyrenew.process import ARProcess, DifferencedProcess


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
        log_rt_rv : RandomVariable
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
        self.ar_diff = DifferencedProcess(
            fundamental_process=ARProcess(
                noise_rv_name=f"{name}{ar_process_suffix}"
            ),
            differencing_order=1,
        )

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
    ) -> ArrayLike:
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
        ArrayLike
            Sampled Rt values.
        """

        # Initial sample
        log_rt_rv = self.log_rt_rv.sample(**kwargs)
        b = self.autoreg_rv.sample(**kwargs)
        s_r = self.periodic_diff_sd_rv.sample(**kwargs)

        # How many periods to sample?
        n_periods = (duration + self.period_size - 1) // self.period_size

        # Running the process

        log_rt = self.ar_diff(
            n=n_periods,
            init_vals=jnp.array([log_rt_rv[0]]),
            autoreg=b,
            noise_sd=s_r,
            fundamental_process_init_vals=jnp.array(
                [log_rt_rv[1] - log_rt_rv[0]]
            ),
        )[0]

        return au.repeat_until_n(
            data=jnp.exp(log_rt),
            n_timepoints=duration,
            offset=self.offset,
            period_size=self.period_size,
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
