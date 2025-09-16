# numpydoc ignore=GL08

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax.typing import ArrayLike

import pyrenew.latent.infection_functions as inf
from pyrenew.metaclass import RandomVariable


class InfectionsSample(NamedTuple):
    """
    A container for holding the output from
    [`Infections.sample`][].

    Attributes
    ----------
    post_initialization_infections:
        The estimated latent infections. Default :obj:`None`.
    """

    post_initialization_infections: ArrayLike | None = None


class Infections(RandomVariable):
    r"""Latent infections

    This class samples infections given $\mathcal{R}(t)$,
    initial infections, and generation interval.

    Notes
    -----
    The mathematical model is given by:

    .. math::

            I(t) = R(t) \times \sum_{\tau < t} I(\tau) g(t-\tau)

    where $I(t)$ is the number of infections at time $t$,
    $R(t)$ is the reproduction number at time $t$, and
    $g(t-\tau)$ is the generation interval.
    """

    @staticmethod
    def validate() -> None:  # numpydoc ignore=GL08
        return None

    def sample(
        self,
        Rt: ArrayLike,
        I0: ArrayLike,
        gen_int: ArrayLike,
        **kwargs,
    ) -> InfectionsSample:
        r"""
        Sample infections given
        $\mathcal{R}(t)$, initial infections,
        and generation interval.

        Parameters
        ----------
        Rt : ArrayLike
            Reproduction number.
        I0 : ArrayLike
            Initial infections vector
            of the same length as the
            generation interval.
        gen_int : ArrayLike
            Generation interval pmf vector.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample calls, should there be any.

        Returns
        -------
        InfectionsSample
            A named tuple with a
            ``post_initialization_infections`` field.
        """
        if I0.shape[0] < gen_int.size:
            raise ValueError(
                "Initial infections vector must be at least as long as "
                "the generation interval. "
                f"Initial infections vector length: {I0.shape[0]}, "
                f"generation interval length: {gen_int.size}."
            )

        if I0.shape[1:] != Rt.shape[1:]:
            raise ValueError(
                "Initial infections and Rt must have the "
                "same batch shapes. "
                f"Got initial infections of batch shape {I0.shape[1:]} "
                f"and Rt of batch shape {Rt.shape[1:]}."
            )

        gen_int_rev = jnp.flip(gen_int)
        recent_I0 = I0[-gen_int_rev.size :]

        post_initialization_infections = inf.compute_infections_from_rt(
            I0=recent_I0,
            Rt=Rt,
            reversed_generation_interval_pmf=gen_int_rev,
        )

        return InfectionsSample(post_initialization_infections)
