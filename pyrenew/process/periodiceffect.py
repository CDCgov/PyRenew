# numpydoc ignore=GL08


import pyrenew.arrayutils as au
from pyrenew.metaclass import RandomVariable


class PeriodicEffect(RandomVariable):
    """
    Periodic effect with repeating values from a random variable.
    """

    def __init__(
        self,
        offset: int,
        quantity_to_broadcast: RandomVariable,
    ):
        """
        Default constructor for PeriodicEffect class.

        Parameters
        ----------
        offset
            Relative point at which data starts, must be between 0 and
            period_size - 1.
        quantity_to_broadcast
            Values to be broadcasted (repeated or tiled).

        Returns
        -------
        None
        """

        PeriodicEffect.validate(quantity_to_broadcast)

        self.offset = offset

        self.quantity_to_broadcast = quantity_to_broadcast

    @staticmethod
    def validate(quantity_to_broadcast: RandomVariable) -> None:
        """
        Validate the broadcasting quatity.

        Parameters
        ----------
        quantity_to_broadcast
            Values to be broadcasted (repeated or tiled).

        Returns
        -------
        None
        """

        assert isinstance(quantity_to_broadcast, RandomVariable)

        return None

    def sample(self, duration: int, **kwargs):
        """
        Sample from the process.

        Parameters
        ----------
        duration
            Number of timepoints to sample.
        **kwargs
            Additional keyword arguments passed through to the `quantity_to_broadcast`.

        Returns
        -------
        ArrayLike
        """

        return au.tile_until_n(
            data=self.quantity_to_broadcast.sample(**kwargs),
            n_timepoints=duration,
            offset=self.offset,
        )


class DayOfWeekEffect(PeriodicEffect):
    """
    Weekly effect with repeating values from a random variable.
    """

    def __init__(
        self,
        offset: int,
        quantity_to_broadcast: RandomVariable,
    ):
        """
        Default constructor for DayOfWeekEffect class.

        Parameters
        ----------
        offset
            Relative point at which data starts, must be between 0 and
            6.
        quantity_to_broadcast
            Values to be broadcasted (repeated or tiled).

        Returns
        -------
        None
        """

        DayOfWeekEffect.validate(offset)

        super().__init__(
            offset=offset,
            quantity_to_broadcast=quantity_to_broadcast,
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
        offset
            Relative point at which data starts, must be between 0 and 6.

        Returns
        -------
        None
        """
        assert isinstance(offset, int), "offset must be an integer."

        assert 0 <= offset <= 6, "offset must be between 0 and 6."

        return None
