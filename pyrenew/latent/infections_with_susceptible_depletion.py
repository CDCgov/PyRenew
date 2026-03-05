# numpydoc ignore=GL08

from typing import NamedTuple

import jax.numpy as jnp
from numpy.typing import ArrayLike

import pyrenew.latent.infection_functions as inf
from pyrenew.metaclass import RandomVariable


class InfectionsSuspDepletionSample(NamedTuple):
    """
    A container for holding the output from the InfectionsWithSusceptibleDepletion.

    Attributes
    ----------
    post_initialization_infections
        The estimated latent infections. Defaults to None.
    rt
        The adjusted reproduction number. Defaults to None.
    """

    post_initialization_infections: ArrayLike | None = None
    rt: ArrayLike | None = None

    def __repr__(self) -> str:
        return f"InfectionsSample(post_initialization_infections={self.post_initialization_infections}, rt={self.rt})"


class InfectionsWithSusceptibleDepletion(RandomVariable):
    r"""
    Latent infections

    This class computes infections, given Rt, initial infections,
    initial susceptible population, and generation interval.

    Parameters
    ----------
    name
        A name for this random variable.

    Notes
    -----
    This function implements the following renewal process with susceptible depletion:

    ```math
    I(t) & = S(t) \left( 1 - \exp\left(\frac{- \mathcal{R}(t) \lambda(t)}{S(t)} \right) \right)

    \lambda(t) & = \sum_{\tau=1}^{T_g}I(t - \tau)g(\tau)
    S(t) & = \max\left(1, S_0 - \sum_{\tau=1}^{t-1} I(\tau)\right)
    ```

    where $\mathcal{R}(t)$ is the reproductive number, $g(t)$
    is the generation interval PMF, $T_g$ is the max-length of the
    generation interval, and $S_0$ is the initial susceptible population.
    """

    def __init__(
        self,
        name: str,
    ) -> None:
        """
        Default constructor for InfectionsWithSusceptibleDepletion class.

        Parameters
        ----------
        name
            A name for this random variable.
        """
        super().__init__(name=name)

    @staticmethod
    def validate() -> None:  # numpydoc ignore=GL08
        return None

    def sample(
        self,
        Rt: ArrayLike,
        I0: ArrayLike,
        gen_int: ArrayLike,
        S0: ArrayLike,
        **kwargs: object,
    ) -> InfectionsSuspDepletionSample:
        """
        Samples infections given Rt, initial infections, and generation
        interval.

        Parameters
        ----------
        Rt
            Reproduction number.
        I0
            Initial infections, as an array
            at least as long as the generation
            interval PMF.
        gen_int
            Generation interval PMF.
        S0
            Initial susceptible population.
        **kwargs
            Additional keyword arguments passed through to internal
            sample calls, should there be any.

        Returns
        -------
        InfectionsSuspDepletionSample
            Named tuple with "infections".
        """
        if I0.shape[0] < gen_int.size:
            raise ValueError(
                "Initial infections must be at least as long as the "
                f"generation interval. Got initial infections length {I0.shape[0]}"
                f"and generation interval length {gen_int.size}."
            )

        if I0.shape[1:] != Rt.shape[1:]:
            raise ValueError(
                "Initial infections and Rt must have the same batch shapes. "
                f"Got initial infections of batch shape {I0.shape[1:]} "
                f"and Rt of batch shape {Rt.shape[1:]}."
            )

        gen_int_rev = jnp.flip(gen_int)

        I0 = I0[-gen_int_rev.size :]

        (
            post_initialization_infections,
            Rt_adj,
        ) = inf.compute_infections_with_susceptible_depletion(
            I0=I0,
            Rt_raw=Rt,
            reversed_generation_interval_pmf=gen_int_rev,
            S0=S0,
        )

        return InfectionsSuspDepletionSample(
            post_initialization_infections=post_initialization_infections,
            rt=Rt_adj,
        )
