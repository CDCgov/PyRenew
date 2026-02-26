"""
Class for vectors that represent discrete probability
mass functions.
"""

from abc import abstractmethod

import jax.numpy as jnp
from jax.typing import ArrayLike

from pyrenew.deterministic import DeterministicVariable
from pyrenew.metaclass import RandomVariable


class PMFVector(RandomVariable):
    """
    Abstract [`pyrenew.metaclass.RandomVariable`][] that
    represents a probability mass function (PMF) as a
    vector of probabilities that sums to 1.

    These vectors of probabilities can be deterministic
    or stochastic in concrete subclasses.
    """

    def __init__(self, name: str, values: ArrayLike, **kwargs) -> None:
        """
        Default constructor.

        Parameters
        ----------
        name
            Name for the random variable.

        values
            Vector of values of the same shape as the
            output of [`self.sample`][], representing the
            values of the variable to which those probabilities
            correspond.

        **kwargs
            Additional keyword arguments passed to the parent
            constructor.

        Returns
        -------
        None
        """
        self.name = name
        self.values = values
        super().__init__(**kwargs)

    @abstractmethod
    def sample(self, **kwargs) -> ArrayLike:
        """
        Sample a vector of probabilities.
        """
        pass


class DelayPMF(PMFVector):
    """
    Subclass of [`pyrenew.randomvariable.PMFVector`] that
    represents a discrete time delay PMF.

    Discrete delay PMFs are fundamental to discrete-time
    renewal modeling. They are used to represent generation
    interval distributions (minimum value 1 time unit), as well
    as delays between infectious and various downstream events
    (e.g. an infection-to-hospital-admission delay distribution,
    minimum value 0 time units).

    Enforces continguousness. [`self.values`][] must be
    an array of consecutive integers representing time units.

    Enforces either 0 or 1 indexing. Shortest represented delay must
    be either 0 or 1 time unit.
    """

    def __init__(self, name: str, min_delay: int, max_delay: int, **kwargs) -> None:
        """
        Default constructor

        Parameters
        ----------
        name
            Name for the random variable.

        min_delay
            Shortest possible delay in time units.
            Will become the first value of [`self.values`][]
            (corresponding to the zeroth entry of the probability
            vector returned by [`self.sample`][]). Must be an integer
            greater than or equal to 0.

        max_delay
            Longest possible delay in time units.
            Will become the final value of [`self.values`][]
            (corresponding to the final entry of the probability
            vector returned by [`self.sample`][]). Must be an
            integer greater than or equal to `min_delay`.

        **kwargs
            Additional keyword arguments passed to the parent
            constructor.

        Returns
        -------
        None

        Raises
        ------
        ValueError
           If min_delay and max_delay do not satisfy the specified
           constraints.
        """
        if not all([isinstance(x, int) for x in [min_delay, max_delay]]):
            raise ValueError("min_delay and max_delay must be integers.")

        if not min_delay > 0:
            raise ValueError("min_delay must be greater than or equal to 0.")
        if not max_delay >= min_delay:
            raise ValueError("max_delay must be greater than or equal to min_delay")

        super().__init__(name=name, values=jnp.arange(min_delay, max_delay + 1))

    @property
    def min_delay(self) -> int:
        """
        The minimum possible delay in integer time units.
        Corresponds to the zeroth entry of the probability vector
        returned by [`self.sample`][].

        Returns
        -------
        int
            The value of the minimum possible delay.
        """
        return self.values[0]

    @property
    def max_delay(self) -> int:
        """
        The maximum possible delay in integer time units.
        Corresponds to the final entry of the probability vector
        returned by [`self.sample`][].

        Returns
        -------
        int
            The value of the maximum possible delay.
        """
        return self.values[-1]


class NonnegativeDelayPMF(DelayPMF):
    """
    Subclass of [`pyrenew.randomvariable.DelayPMF`] that
    represents the PMF of a delay that can possibly be
    0 time units (i.e. no delay).

    Enforces a `min_delay` value of 0.

    In PyRenew, we have a convention of using
    `NonnegativeDelayPMF`s to represent discrete-time delays
    from infection to ascertained observation. This
    simplifies the computation of predicted observations.
    """

    def __init__(self, name: str, max_delay: int) -> None:
        """
        Default constructor.

        Parameters
        ----------
        name
            Name for the random variable.

        max_delay
            Longest possible delay in time units.
            Will become the final value of [`self.values`][]
            (corresponding to the final entry of the probability
            vector returned by [`self.sample`][]). Must be an
            integer greater than or equal to 0.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If max_delay does not satisfy the specified constraints.
        """
        super().__init__(name=name, min_delay=0, max_delay=max_delay)


class PositiveDelayPMF(DelayPMF):
    """
    Subclass of [`pyrenew.randomvariable.DelayPMF`] that
    represents the PMF of a strictly positive discrete time
    delay (i.e. of at least 1 time unit).

    Enforces a `min_delay` value of 1.

    In PyRenew, we have a convention of using
    `PositiveDelayPMF`s to represent generation interval
    distributions. This simplifies the computation of the
    renewal equation.
    """

    def __init__(self, name: str, max_delay: int) -> None:
        """
        Default constructor.

        Parameters
        ----------
        name
            Name for the random variable.

        max_delay
            Longest possible delay in time units.
            Will become the final value of [`self.values`][]
            (corresponding to the final entry of the probability
            vector returned by [`self.sample`][]). Must be an
            integer greater than or equal to 1.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If max_delay does not satisfy the specified constraints.
        """

        super().__init__(name=name, min_delay=1, max_delay=max_delay)


class GenerationIntervalPMF(PositiveDelayPMF):
    """
    Subclass of [`pyrenew.randomvariable.PositiveDelayPMF`] that
    represents the PMF of a generation interval distribution.
    """


class AscertainmentDelayPMF(NonnegativeDelayPMF):
    """
    Subclass of [`pyrenew.randomvariable.NonnegativeDelayPMF`] that
    represents the PMF of a delay from an event to when it is
    ascertained
    """


class DeterministicGenerationIntervalPMF(GenerationIntervalPMF):
    """
    Subclass of [`pyrenew.randomvariable.GenerationIntervalPMF`]
    where the PMF is treated as fixed.
    """

    def __init__(self, name: str, probabilities: ArrayLike, max_delay: int) -> None:
        """
        Default constructor.

        Parameters
        ----------
        name
            Name for the random variable.

        probabilities
            Vector of probabilities representing the pmf

        max_delay
            Longest possible delay in time units.
            Will become the final value of [`self.values`][]
            (corresponding to the final entry of the probability
            vector returned by [`self.sample`][]). Must be an
            integer greater than or equal to 1.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If max_delay does not satisfy the specified constraints.
        """

        self.base_variable_ = DeterministicVariable(
            name="base_variable_", value=probabilities
        )
        super().__init__(name=name, max_delay=max_delay)

    def sample(self, **kwargs) -> ArrayLike:
        """
        Retrieve the probability vector representing
        the deterministic PMF.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to `self.base_variable_.sample()`.

        Returns
        -------
        ArrayLike
            The probability vector.
        """
        return self.base_variable_.sample(**kwargs)
