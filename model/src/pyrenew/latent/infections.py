# -*- coding: utf-8 -*-

from collections import namedtuple

import jax.numpy as jnp
import numpyro as npro
import pyrenew.latent.infection_functions as inf
from numpy.typing import ArrayLike
from pyrenew.metaclass import RandomVariable

InfectionsSample = namedtuple(
    "InfectionsSample",
    ["infections"],
    defaults=[None],
)


class Infections(RandomVariable):
    """Latent infections"""

    def __init__(
        self,
        infections_mean_varname: str = "latent_infections",
    ) -> None:
        """Default constructor

        Parameters
        ----------
        infections_mean_varname : str.
            Name of the element in `random_variables` that will hold the value
            of mean 'infections'.

        Returns
        -------
        None
        """

        self.infections_mean_varname = infections_mean_varname

        return None

    @staticmethod
    def validate() -> None:
        return None

    def sample(
        self,
        Rt: ArrayLike,
        I0: ArrayLike,
        gen_int: ArrayLike,
        **kwargs,
    ) -> tuple:
        """Samples infections given Rt

        A random variable representing the pmf of the generation interval.

        Parameters
        ----------
        Rt : ArrayLike
            Reproduction number.
        I0 : ArrayLike
            Initial infections.
        gen_int : ArrayLike
            Generation interval.

        Returns
        -------
        InfectionsSample
            Named tuple with "infections".
        """

        gen_int_rev = jnp.flip(gen_int)

        n_lead = gen_int_rev.size - 1
        I0_vec = jnp.hstack([jnp.zeros(n_lead), I0])

        all_infections = inf.sample_infections_rt(
            I0=I0_vec,
            Rt=Rt,
            reversed_generation_interval_pmf=gen_int_rev,
        )

        npro.deterministic(self.infections_mean_varname, all_infections)

        return InfectionsSample(all_infections)
