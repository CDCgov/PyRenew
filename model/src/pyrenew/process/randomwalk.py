# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan
from pyrenew.metaclass import DistributionalRV, RandomVariable, SampledValue
from pyrenew.process.differencedprocess import DifferencedProcess


class IIDRamdomSequence(RandomVariable):
    """
    Class for constructing random sequence with
    independent and identically distributed elements
    and an arbitrary next element distribution.
    """

    def __init__(
        self,
        name: str,
        element_dist: dist.Distribution,
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
        element_dist : Distribution
            numpyro.distributions.Distribution representing
            the distribution of elements in the sequence.

        Returns
        -------
        None
        """
        super().__init__(**kwargs)
        self.name = name
        self.element_dist = element_dist

    def sample(self, n_elements: int, vectorize: bool = False) -> tuple:
        """
        Sample an IID random sequence.

        Parameters
        ----------
        n_elements : int
            Length of the sequence to sample.

        vectorize: bool
            Sample vectorized? If True, use
            :meth:`dist.Distribution.expand()`,
            if False use
            :fun:`numpyro.contrib.control_flow.scan`.
            Default False.

        Returns
        -------
        SampledValue
            Whose value is an array of n_elements
            samples from :meth:`self.element_rv.sample()`
        """

        if not vectorize:

            def transition(x_prev, _):
                # numpydoc ignore=GL08
                el = numpyro.sample(self.name, self.element_dist)
                return el, el

            _, result = scan(
                transition,
                init=jnp.zeros(self.element_dist.batch_shape),
                xs=jnp.arange(n_elements),
            )
        else:
            result = numpyro.sample(
                self.name, self.element_dist.expand((n_elements,))
            )

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


class RandomWalk(RandomVariable):
    """
    Class for a Markovian
    random walk with an arbitrary
    step distribution
    """

    def __init__(
        self,
        name: str,
        step_rv: RandomVariable,
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

        step_rv : RandomVariable
            RandomVariable representing the step distribution.

        **kwargs :
            Additional keyword arguments passed to the parent
            class constructor.

        Returns
        -------
        None
        """
        super().__init__(**kwargs)
        self.name = name
        self.step_rv = step_rv

    def sample(
        self,
        n_steps: int,
        init_val: float,
        **kwargs,
    ) -> tuple:
        """
        Sample from the random walk.

        Parameters
        ----------
        n_steps : int
            Length of the walk to sample.
        init_val : float
            Initial value of the walk.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, should there be any.

        Returns
        -------
        tuple
            With a single array of shape (n_steps,).
        """

        def transition(x_prev, _):
            # numpydoc ignore=GL08
            diff, *_ = self.step_rv(**kwargs)
            x_curr = x_prev + diff.value
            return x_curr, x_curr

        _, x = scan(
            transition,
            init=init_val,
            xs=jnp.arange(n_steps - 1),
        )

        return (
            SampledValue(
                jnp.hstack([init_val, x.flatten()]),
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


class StandardNormalRandomWalk(RandomWalk):
    """
    A random walk with
    standard Normal (mean = 0, standard deviation = 1)
    steps, implemented via the base RandomWalk class.
    """

    def __init__(self, name: str, step_suffix: str = "_step", **kwargs):
        """
        Default constructor

        Parameters
        ----------
        name : str
            A name for the random variable.

        step_suffix : str
           A suffix to append to the name when
           sampling the random walk steps via
           numpyro.sample (the name of the site/
           parameter in numpyro will be
           self.name + self.step_suffix.

        **kwargs:
            Additional keyword arguments passed
            to the parent class constructor.

        Returns
        -------
        None
        """
        super().__init__(
            name=name,
            step_rv=DistributionalRV(
                name=name + step_suffix, dist=dist.Normal(loc=0, scale=1)
            ),
            **kwargs,
        )


class RandomWalk2(DifferencedProcess):
    """
    Alternative class for a Markovian
    random walk with an arbitrary
    step distribution, implemented
    via DifferencedProcess and IIDRamdomSequence
    """

    def __init__(
        self,
        name: str,
        step_distribution: dist.Distribution,
        step_suffix="_step",
        **kwargs,
    ):
        """
                Default constructor

        Parameters
        ----------
        name : str
            A name for the random variable, used to
            name sites within it in :fun:`numpyro.sample()`
            calls.

        step_distribution : dist.Distribution
            numpyro.distributions.Distribution
            representing the step distribution
            of the random walk.

        step_suffix : str
            Suffix to append to the RandomVariable's
            name when naming the random variable that
            holds its steps (differences). Default
            "_step".

        **kwargs :
            Additional keyword arguments passed to the parent
            class constructor.

        Returns
        -------
        None
        """
        super().__init__(
            name=name,
            fundamental_process=IIDRamdomSequence(
                name=name + step_suffix, element_dist=step_distribution
            ),
            differencing_order=1,
            **kwargs,
        )
