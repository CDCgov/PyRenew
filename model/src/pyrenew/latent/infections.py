# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpyro as npro
import pyrenew.latent.infection_functions as inf
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable


class InfectionsSample(NamedTuple):
    """
    A container for holding the output from `latent.Infections.sample()`.

    Attributes
    ----------
    post_seed_infections : ArrayLike | None, optional
        The estimated latent infections. Defaults to None.
    """

    post_seed_infections: ArrayLike | None = None

    def __repr__(self):
        return f"InfectionsSample(infections={self.infections})"


class Infections(RandomVariable):
    r"""Latent infections

    This class samples infections given Rt,
    initial infections, and generation interval.

    Notes
    -----
    The mathematical model is given by:

    .. math::

            I(t) = R(t) \times \sum_{\tau < t} I(\tau) g(t-\tau)

    where :math:`I(t)` is the number of infections at time :math:`t`,
    :math:`R(t)` is the reproduction number at time :math:`t`, and
    :math:`g(t-\tau)` is the generation interval.
    """

    def __init__(
        self,
        latent_infections_varname: str,
    ) -> None:
        """
        Default constructor for Infections class.

        Parameters
        ----------
        latent_infections_varname : str
            Name to be assigned to the deterministic variable in the model.

        Returns
        -------
        None
        """

        self.latent_infections_varname = latent_infections_varname

        return None

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
        """
        Samples infections given Rt, initial infections, and generation
        interval.

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
            Named tuple with "infections".
        """
        if I0.size < gen_int.size:
            raise ValueError(
                "Initial infections vector must be at least as long as "
                "the generation interval. "
                f"Initial infections vector length: {I0.size}, "
                f"generation interval length: {gen_int.size}."
            )

        gen_int_rev = jnp.flip(gen_int)
        recent_I0 = I0[-gen_int_rev.size :]

        post_seed_infections = inf.compute_infections_from_rt(
            I0=recent_I0,
            Rt=Rt,
            reversed_generation_interval_pmf=gen_int_rev,
        )

        npro.deterministic(
            self.latent_infections_varname, post_seed_infections
        )
        return InfectionsSample(post_seed_infections)
