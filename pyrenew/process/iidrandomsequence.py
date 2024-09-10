# numpydoc ignore=GL08

import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.contrib.control_flow import scan

from pyrenew.metaclass import RandomVariable
from pyrenew.randomvariable import DistributionalVariable


class IIDRandomSequence(RandomVariable):
    """
    Class for constructing random sequence of
    independent and identically distributed elements
    given an arbitrary RandomVariable representing
    those elements.
    """

    def __init__(
        self,
        element_rv: RandomVariable,
        **kwargs,
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        name : str
            A name for the random variable, used to
            name sites within it in :meth:`numpyro.sample()`
            calls.
        element_rv : RandomVariable
            RandomVariable representing a single element
            in the sequence.

        Returns
        -------
        None
        """
        super().__init__(**kwargs)
        self.element_rv = element_rv

    def sample(
        self, n: int, *args, vectorize: bool = False, **kwargs
    ) -> ArrayLike:
        """
        Sample an IID random sequence.

        Parameters
        ----------
        n : int
            Length of the sequence to sample.

        *args :
            Additional positional arguments passed
            to self.element_rv.sample()

        vectorize: bool
            Sample vectorized? If True, use
            :meth:`RandomVariable.expand_by()`,
            whenever available, and fall back on
            :meth:`numpyro.contrib.control_flow.scan`.
            If False, always use :meth:`scan()`.
            Default False.

        **kwargs:
            Additional keyword arguments passed to
            self.element_rv.sample().

        Returns
        -------
        ArrayLike
            `n` samples from `self.distribution`
        """

        if vectorize and hasattr(self.element_rv, "expand_by"):
            result, *_ = self.element_rv.expand_by((n,)).sample(
                *args, **kwargs
            )
            result = result
        else:

            def transition(_carry, _x):
                # numpydoc ignore=GL08
                el, *_ = self.element_rv.sample(*args, **kwargs)
                return None, el

            _, result = scan(
                transition,
                xs=None,
                init=None,
                length=n,
            )

        return result

    @staticmethod
    def validate():
        """
        Validates input parameters, implementation pending.
        """
        super().validate()
        return None


class StandardNormalSequence(IIDRandomSequence):
    """
    Class for a sequence of IID standard Normal
    (mean = 0, sd = 1) random variables.
    """

    def __init__(
        self,
        element_rv_name: str,
        **kwargs,
    ):
        """
        Default constructor

        Parameters
        ----------
        name : str
            see :class:`IIDRandomSequence`.
        element_rv_name: str
            Name for the internal element_rv, here a
            DistributionalVariable encoding a
            standard Normal (mean = 0, sd = 1)
            distribution.

        Returns
        -------
        None
        """
        super().__init__(
            element_rv=DistributionalVariable(
                name=element_rv_name, distribution=dist.Normal(0, 1)
            ),
        )
