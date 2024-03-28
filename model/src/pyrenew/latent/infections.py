# -*- coding: utf-8 -*-

from collections import namedtuple

import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
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
        gen_int: RandomVariable,
        I0_varname: str = "I0",
        Rt_varname: str = "Rt",
        infections_mean_varname: str = "latent_infections",
        I0_dist: dist.Distribution = dist.LogNormal(2, 0.25),
    ) -> None:
        """Default constructor

        Parameters
        ----------
        gen_int: RandomVariable
            A random variable representing the pmf of the generation interval.
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
        Infections.validate(I0_dist, gen_int)

        self.I0_dist = I0_dist
        self.gen_int_rev = gen_int

        self.I0_varname = I0_varname
        self.Rt_varname = Rt_varname
        self.infections_mean_varname = infections_mean_varname

        return None

    @staticmethod
    def validate(I0_dist, gen_int) -> None:
        assert isinstance(I0_dist, dist.Distribution)
        assert isinstance(gen_int, RandomVariable)

        return None

    def sample(
        self,
        random_variables: dict,
        constants: dict = None,
    ) -> tuple:
        """Samples infections given Rt

        Parameters
        ----------
        obs : random_variables, optional
            A dictionary containing an observed `Rt` sequence passed to
            `sample_infections_rt()`. It can also contain `infections` and `I0`,
            both passed to `obs` in `numpyro.sample()`.
        constants : dict
            Possible dictionary of constants.

        Returns
        -------
        InfectionsSample
            Named tuple with "infections".
        """
        I0 = npro.sample(
            name="I0",
            fn=self.I0_dist,
            obs=random_variables.get(self.I0_varname, None),
        )

        gen_int_rev, *_ = self.gen_int_rev.sample(
            random_variables=random_variables,
            constants=constants,
        )

        n_lead = gen_int_rev.size - 1
        I0_vec = jnp.hstack([jnp.zeros(n_lead), I0])

        all_infections = inf.sample_infections_rt(
            I0=I0_vec,
            Rt=random_variables.get(self.Rt_varname),
            reversed_generation_interval_pmf=gen_int_rev,
        )

        npro.deterministic(self.infections_mean_varname, all_infections)

        return InfectionsSample(all_infections)
