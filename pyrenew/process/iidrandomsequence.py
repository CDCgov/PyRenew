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
        name: str,
        element_rv: RandomVariable,
        **kwargs: object,
    ) -> None:
        """
        Default constructor.

        Parameters
        ----------
        name
            A name for this random variable.
        element_rv
            RandomVariable representing a single element
            in the sequence.

        Returns
        -------
        None
        """
        super().__init__(name=name, **kwargs)
        self.element_rv = element_rv

    def sample(self, n: int, *args: object, vectorize: bool = False, **kwargs: object) -> ArrayLike:
        """
        Sample an IID random sequence.

        Parameters
        ----------
        n
            Length of the sequence to sample.

        *args
            Additional positional arguments passed
            to self.element_rv.sample()

        vectorize
            Sample vectorized? If True, use the
            [`pyrenew.metaclass.RandomVariable`][]'s
            `expand_by()` method, if available,
            and fall back on [`numpyro.contrib.control_flow.scan`][]
            otherwise.
            If False, always use
            [`numpyro.contrib.control_flow.scan`][].
            Default False.

        **kwargs
            Additional keyword arguments passed to
            `self.element_rv.sample`.

        Returns
        -------
        ArrayLike
            `n` samples from self.distribution`.
        """

        if vectorize and hasattr(self.element_rv, "expand_by"):
            result = self.element_rv.expand_by((n,)).sample(*args, **kwargs)
        else:

            def transition(_carry: None, _x: None) -> tuple[None, ArrayLike]:
                # numpydoc ignore=GL08
                el = self.element_rv.sample(*args, **kwargs)
                return None, el

            _, result = scan(
                transition,
                xs=None,
                init=None,
                length=n,
            )

        return result

    @staticmethod
    def validate() -> None:
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
        name: str,
        element_shape: tuple = None,
        **kwargs: object,
    ) -> None:
        """
        Default constructor.

        Parameters
        ----------
        name
            A name for this random variable.
            The internal element distribution is named
            ``f"{name}_element"``.
        element_shape
            Shape for each element in the sequence.
            If None, elements are scalars. Default
            None.

        Returns
        -------
        None
        """
        if element_shape is None:
            element_shape = ()
        super().__init__(
            name=name,
            element_rv=DistributionalVariable(
                name=f"{name}_element", distribution=dist.Normal(0, 1)
            ).expand_by(element_shape),
        )
