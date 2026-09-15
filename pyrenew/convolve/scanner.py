"""
Scanner classes for infection generation processes.
Class instances are callables that can be passed as the
`fn` argument to [`jax.lax.scan`][] or
[`numpyro.contrib.control_flow.scan`][].
"""

from abc import ABCMeta, abstractmethod
from collections.abc import Callable

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import PyTree


def _next_carry_array_append(new_value: ArrayLike, old_carry: ArrayLike) -> ArrayLike:
    """
    Create a new carry Array from an old array and a new latest value
    by appending the new (latest) value to the end of the array.

    Carry arrays produced by this function move forward in time: the
    oldest values are at the beginning, the latest are at the end.
    Arrays grow in length at every iteration.

    Follows PyRenew and scan conventions for multi-dimensional arrays
    by appending along the leading axis.

    Parameters
    ----------
    new_value
        Value to append.

    old_carry
        Previous carry vector.

    Returns
    -------
    new_carry:
        New carry array produced by and appending `new_value`
        to the end of `old_carry`.
    """
    return jnp.concatenate([old_carry, new_value[jnp.newaxis]], axis=0)


def _next_carry_array_sliding_window(
    new_value: ArrayLike, old_carry: ArrayLike
) -> ArrayLike:
    """
    Create a new carry Array from an old array and a new latest value
    by dropping the first (oldest) value and appending the new (latest)
    value to the end of the array.

    Carry arrays produced by this function move forward in time: the
    oldest values are at the beginning, the latest are at the end.
    Arrays remain the same length for all iterations.

    Follows PyRenew and scan conventions for multi-dimensional arrays
    by appending along the leading axis.

    Parameters
    ----------
    new_value
        Value to append.

    old_carry
        Previous carry vector.

    Returns
    -------
    new_carry:
        New carry array produced by dropping the first entry in
        `old_carry` and appending `new_value` as the last entry.
    """
    return jnp.concatenate([old_carry[1:], new_value[jnp.newaxis]], axis=0)


class Scanner(metaclass=ABCMeta):
    """
    Abstract base class for scanner callables.
    """

    @abstractmethod
    def __call__(self, carry: PyTree, x: PyTree) -> tuple[PyTree, PyTree]:
        """
        Template call method with signature compatible with
        [`jax.lax.scan`][], [`numpyro.contrib.control_flow.scan`][]
        and similar.
        """
        raise NotImplementedError()


class BaseBackwardLookingConvolutionScanner(Scanner):
    """
    Generic class for single fixed-array backward-looking convolution:

    1. A fixed array is convolved along its leading
    axis with the existing carry, producing a value (`convolved`).
    2. an arbitrary user-specified function `f` is applied to the
    current value (`x`) of the PyTree being scanned
    along and the convolution result (`convolved`), with the call
    signature `f(x, convolved)`.

    3. The result of `f(x, convolved)` is the `latest` value of the scan,
    and a new carry of equal length to the old carry.

    This generic class should not generally be instantiated as-is,
    but rather to produce more specific subclasses with meaningful
    choices of the function `f`.
    """

    def __init__(self, array_to_convolve: ArrayLike, f: Callable) -> None:
        """
        Constructor.

        Parameters
        ----------
        array_to_convolve
            Array to convolve with the carry at
            each iterative step.

        f
            Function to apply to the convolution result
            to generate the latest value in the iteration,
            which also becomes the latest value in the new
            carry.
        """

        self.array = array_to_convolve
        self.f = f

    def __call__(self, carry: ArrayLike, x: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """
        Compute the latest value and new carry.

        Parameters
        ----------
        carry
            Current value of the carry

        x
            Current entry of the array being scanned over.

        Returns
        -------
        latest, new_carry:
            The latest generated value and the value of the new carry.

        """
        convolved = jnp.einsum("i...,i...->...", self.array, carry)
        latest = self.f(x, convolved)
        new_carry = _next_carry_array_sliding_window(latest, carry)

        return latest, new_carry


class ConvolveAndMultiplyScanner(BaseBackwardLookingConvolutionScanner):
    r"""
    Scanner that computes

    ```math
    Y(t) = f\left(x(t) \begin{bmatrix} Y(t - n) \\ Y(t - n + 1) \\
    \vdots{} \\ Y(t - 1)\end{bmatrix} \cdot{} \mathbf{d} \right)
    ```

    where $\mathbf{d}$ is an array of length $n$ given by `array_to_convolve`
    and $x(t)$ is the current value of the array being scanned along.
    """

    def __init__(self, array_to_convolve: ArrayLike) -> None:
        """
        Constructor

        Parameters
        ----------
        array_to_convolve
            Fixed array to convolve with the carry at
            each step of iteration.
        """
        super().__init__(
            array_to_convolve=array_to_convolve, f=lambda x, convolved: x * convolved
        )
