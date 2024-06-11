# numpydoc ignore=GL08

from typing import NamedTuple

import jax.numpy as jnp
import pyrenew.arrayutils as au
from pyrenew.metaclass import RandomVariable, _assert_sample_and_rtype


class PeriodicEffectSample(NamedTuple):
    """
    A container for holding the output from
    `process.PeriodicEffect.sample()`.

    Attributes
    ----------
    value: jnp.ndarray
        The sampled value.
    """

    value: jnp.ndarray

    def __repr__(self):
        return f"PeriodicEffectSample(value={self.value})"


class PeriodicEffect(RandomVariable):
    """
    Periodic effect with repeating values from a prior.
    """

    def __init__(
        self,
        data_starts: int,
        period_size: int,
        prior: RandomVariable,
    ):
        """
        Default constructor for PeriodicEffect class.

        Parameters
        ----------
        data_starts : int
            Relative point at which data starts, must be between 0 and
            period_size - 1.
        period_size : int
            Size of the period.
        prior : RandomVariable
            Prior distribution.

        Returns
        -------
        None
        """

        PeriodicEffect.validate(prior)

        self.broadcaster = au.PeriodicBroadcaster(
            data_starts=data_starts,
            period_size=period_size,
            broadcast_type="tile",
        )

        self.prior = prior

    @staticmethod
    def validate(prior: RandomVariable) -> None:
        """
        Validate the prior.

        Parameters
        ----------
        prior : RandomVariable
            Prior distribution.

        Returns
        -------
        None
        """

        _assert_sample_and_rtype(prior)

        return None

    def sample(self, duration: int, **kwargs):
        """
        Sample from the process.

        Parameters
        ----------
        duration : int
            Number of timepoints to sample.
        **kwargs : dict, optional
            Additional keyword arguments passed through to the prior.

        Returns
        -------
        PeriodicEffectSample
        """

        return PeriodicEffectSample(
            value=self.broadcaster(
                data=self.prior.sample(**kwargs)[0],
                n_timepoints=duration,
            )
        )


class DayOfWeekEffect(PeriodicEffect):
    """
    Weekly effect with repeating values from a prior.
    """

    def __init__(
        self,
        data_starts: int,
        prior: RandomVariable,
    ):
        """
        Default constructor for DayOfWeekEffect class.

        Parameters
        ----------
        data_starts : int
            Relative point at which data starts, must be between 0 and
            6.
        prior : RandomVariable
            Prior distribution.

        Returns
        -------
        None
        """

        DayOfWeekEffect.validate(data_starts)

        super().__init__(data_starts=data_starts, period_size=7, prior=prior)

        return None

    @staticmethod
    def validate(
        data_starts: int,
    ):
        """
        Validate the input parameters.

        Parameters
        ----------
        data_starts : int
            Relative point at which data starts, must be between 0 and 6.

        Returns
        -------
        None
        """
        assert isinstance(data_starts, int), "data_starts must be an integer."

        assert 0 <= data_starts <= 6, "data_starts must be between 0 and 6."

        return None
