# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import jax.numpy as jnp
from numpyro.contrib.control_flow import scan
from pyrenew.metaclass import RandomVariable, SampledValue


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
        if not hasattr(element_rv, "sample_shape"):
            raise ValueError(
                "Only RandomVariables with a sample_shape()"
                "method can be used as the element_rv for "
                "an IIDRamdomSequence"
            )

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
        SampledValue
            Whose value is an array of `n`
            samples from `self.distribution`
        """

        if not vectorize:

            def transition(x_prev, _):
                # numpydoc ignore=GL08
                el, *_ = self.element_rv.sample(*args, **kwargs)
                return el.value, el.value

            _, result = scan(
                transition,
                init=jnp.zeros(self.element_rv.batch_shape),
                xs=jnp.arange(n),
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
