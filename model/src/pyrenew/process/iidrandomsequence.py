# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan
from pyrenew.metaclass import DistributionalRV, RandomVariable, SampledValue


class IIDRamdomSequence(RandomVariable):
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
        **kwargs,
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        name : str
            A name for the random variable, used to
            name sites within it in :fun:`numpyro.sample()`
            calls.
        element_rv : RandomVariable
            RandomVariable representing a single element
            in the sequence.

        Returns
        -------
        None
        """
        super().__init__(**kwargs)
        self.name = name
        self.element_rv = element_rv

    def sample(
        self, n: int, *args, vectorize: bool = False, **kwargs
    ) -> tuple:
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
            :meth:`RandomVariable.expand()`,
            if False use
            :fun:`numpyro.contrib.control_flow.scan`.
            Default False.

        **kwargs:
            Additional keyword arguments passed to
            self.element_rv.sample().

        Returns
        -------
        tuple[SampledValue]
            Whose value is an array of `n`
            samples from `self.distribution`
        """

        if not vectorize:

            def transition(_carry, _x):
                # numpydoc ignore=GL08
                el, *_ = self.element_rv.sample(*args, **kwargs)
                return None, el.value

            _, result = scan(
                transition,
                xs=None,
                init=None,
                length=n,
            )
        else:
            result = self.element_rv.expand((n,)).sample(*args, **kwargs)

        return (
            SampledValue(
                result,
                t_start=self.t_start,
                t_unit=self.t_unit,
            ),
        )

    @staticmethod
    def validate():
        """
        Validates input parameters, implementation pending.
        """
        super().validate()
        return None


class StandardNormalSequence(IIDRamdomSequence):
    """
    Class for a sequence of IID standard Normal
    (mean = 0, sd = 1) random variables.
    """

    def __init__(
        self,
        name: str,
        element_suffix: str = "_standard_normal_element",
        **kwargs,
    ):
        """
        Default constructor

        Parameters
        ----------
        name : str
            see :class:`IIDRamdomSequence`.
        element_suffix: str
            Suffix appended to name to name
            the internal element_rv, here a
            DistributionalRV encoding a
            standard Normal (mean = 0, sd = 1)
            distribution. Default "_standard_normal_element"

        Returns
        -------
        None
        """
        super().__init__(
            name=name,
            element_rv=DistributionalRV(
                name=name + element_suffix, distribution=dist.Normal(0, 1)
            ),
        )
