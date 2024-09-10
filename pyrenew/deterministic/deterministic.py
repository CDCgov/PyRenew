# numpydoc ignore=GL08

from __future__ import annotations

import numpyro
from jax.typing import ArrayLike

from pyrenew.metaclass import RandomVariable


class DeterministicVariable(RandomVariable):
    """
    A deterministic (degenerate) random variable. Useful to pass fixed
    quantities.
    """

    def __init__(
        self,
        name: str,
        value: ArrayLike,
        t_start: int | None = None,
        t_unit: int | None = None,
    ) -> None:
        """Default constructor

        Parameters
        ----------
        name : str
            A name to assign to the variable.
        value : ArrayLike
            An ArrayLike object.
        t_start : int, optional
            The start time of the variable, if any.
        t_unit : int, optional
            The unit of time relative to the model's fundamental (smallest) time unit, if any

        Returns
        -------
        None
        """
        self.name = name
        self.validate(value)
        self = value
        self.set_timeseries(t_start, t_unit)

        return None

    @staticmethod
    def validate(value: ArrayLike) -> None:
        """
        Validates input to DeterministicVariable

        Parameters
        ----------
        value : ArrayLike
            An ArrayLike object.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If the input value object is not an ArrayLike object.
        """
        if not isinstance(value, ArrayLike):
            raise ValueError(
                f"value {value} passed to a DeterministicVariable "
                f"is of type {type(value).__name__}, expected "
                "an ArrayLike object"
            )

        return None

    def sample(
        self,
        record=False,
        **kwargs,
    ) -> ArrayLike:
        """
        Retrieve the value of the deterministic Rv

        Parameters
        ----------
        record : bool, optional
            Whether to record the value of the deterministic
            RandomVariable. Defaults to False.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample calls, should there be any.

        Returns
        -------
        ArrayLike
        """
        if record:
            numpyro.deterministic(self.name, self)
        return self
