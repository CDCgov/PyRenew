# -*- coding: utf-8 -*-

from collections import namedtuple

import jax.numpy as jnp
import numpyro as npro
import pyrenew.latent.infection_functions as inf
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
        gen_int_varname: str = "gen_int",
        I0_varname: str = "I0",
        Rt_varname: str = "Rt",
        infections_mean_varname: str = "latent_infections",
    ) -> None:
        """Default constructor

        Parameters
        ----------
        gen_int_varname : str.
            Name of the element in `random_variables` that will hold the value
            of 'generation interval'.
        I0_varname : str.
            Name of the element in `random_variables` that will hold the value
            of 'I0'.
        Rt_varname : str.
            Name of the element in `random_variables` that will hold the value
            of 'Rt'.
        infections_mean_varname : str.
            Name of the element in `random_variables` that will hold the value
            of mean 'infections'.
        I0_dist : dist.Distribution, optional
            Distribution from where to sample the baseline number of infections.

        Returns
        -------
        None
        """

        self.gen_int_varname = gen_int_varname
        self.I0_varname = I0_varname
        self.Rt_varname = Rt_varname
        self.infections_mean_varname = infections_mean_varname

        return None

    @staticmethod
    def validate() -> None:
        return None

    def sample(
        self,
        random_variables: dict,
        constants: dict = None,
    ) -> tuple:
        """Samples infections given Rt

        A random variable representing the pmf of the generation interval.

        Parameters
        ----------
        random_variables : dict
            A dictionary containing an observed `Rt` sequence passed to
            `sample_infections_rt()`. It can also contain `infections` and `I0`,
            both passed to `obs` in `numpyro.sample()`.
        constants : dict, optional
            Ignored.

        Returns
        -------
        InfectionsSample
            Named tuple with "infections".
        """

        I0 = random_variables.get(self.I0_varname)

        gen_int_rev = jnp.flip(random_variables.get(self.gen_int_varname))

        n_lead = gen_int_rev.size - 1
        I0_vec = jnp.hstack([jnp.zeros(n_lead), I0])

        all_infections = inf.sample_infections_rt(
            I0=I0_vec,
            Rt=random_variables.get(self.Rt_varname),
            reversed_generation_interval_pmf=gen_int_rev,
        )

        npro.deterministic(self.infections_mean_varname, all_infections)

        return InfectionsSample(all_infections)
