# numpydoc ignore=GL08

from typing import NamedTuple

import jax.numpy as jnp
import pyrenew.arrayutils as au
from pyrenew.metaclass import RandomVariable, _assert_sample_and_rtype, TimeArray


class PeriodicEffectSample(NamedTuple):
    """
    A container for holding the output from
    `process.PeriodicEffect()`.

    Attributes
    ----------
    value: TimeArray
        The sampled value.
    """

    value: jnp.ndarray

    def __repr__(self):
        return f"PeriodicEffectSample(value={self.value})"


class PeriodicEffect(RandomVariable):
    """
    Periodic effect with repeating values from a random variable.
    """

    def __init__(
        self,
        offset: int,
        period_size: int,
        quantity_to_broadcast: RandomVariable,
        t_start: int,
        t_unit: int,
    ):
        """
        Default constructor for PeriodicEffect class.

        Parameters
        ----------
        offset : int
            Relative point at which data starts, must be between 0 and
            period_size - 1.
        period_size : int
            Size of the period.
        quantity_to_broadcast : RandomVariable
            Values to be broadcasted (repeated or tiled).
        t_start : int
            Start time of the process.
        t_unit : int
            Unit of time relative to the model's fundamental (smallest) time unit.

        Returns
        -------
        None
        """

        PeriodicEffect.validate(quantity_to_broadcast)

        self.broadcaster = au.PeriodicBroadcaster(
            offset=offset,
            period_size=period_size,
            broadcast_type="tile",
        )

        self.set_timeseries(
            t_start=t_start,
            t_unit=t_unit,
        )

        self.quantity_to_broadcast = quantity_to_broadcast

    @staticmethod
    def validate(quantity_to_broadcast: RandomVariable) -> None:
        """
        Validate the broadcasting quatity.

        Parameters
        ----------
        quantity_to_broadcast : RandomVariable
            Values to be broadcasted (repeated or tiled).

        Returns
        -------
        None
        """

        _assert_sample_and_rtype(quantity_to_broadcast)

        return None

    def sample(self, duration: int, **kwargs):
        """
        Sample from the process.

        Parameters
        ----------
        duration : int
            Number of timepoints to sample.
        **kwargs : dict, optional
            Additional keyword arguments passed through to the `quantity_to_broadcast`.

        Returns
        -------
        PeriodicEffectSample
        """

        return PeriodicEffectSample(
            value=TimeArray(self.broadcaster(
                data=self.quantity_to_broadcast.sample(**kwargs)[0].array,
                n_timepoints=duration,
            ))
        )


class DayOfWeekEffect(PeriodicEffect):
    """
    Weekly effect with repeating values from a random variable.
    """

    def __init__(
        self,
        offset: int,
        quantity_to_broadcast: RandomVariable,
        t_start: int,
    ):
        """
        Default constructor for DayOfWeekEffect class.

        Parameters
        ----------
        offset : int
            Relative point at which data starts, must be between 0 and
            6.
        quantity_to_broadcast : RandomVariable
            Values to be broadcasted (repeated or tiled).
        t_start : int
            Start time of the process.

        Returns
        -------
        None
        """

        DayOfWeekEffect.validate(offset)

        super().__init__(
            offset=offset,
            period_size=7,
            quantity_to_broadcast=quantity_to_broadcast,
            t_start=t_start,
            t_unit=1,
        )

        return None

    @staticmethod
    def validate(
        offset: int,
    ):
        """
        Validate the input parameters.

        Parameters
        ----------
        offset : int
            Relative point at which data starts, must be between 0 and 6.

        Returns
        -------
        None
        """
        assert isinstance(offset, int), "offset must be an integer."

        assert 0 <= offset <= 6, "offset must be between 0 and 6."

        return None
