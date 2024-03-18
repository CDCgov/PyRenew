from collections import namedtuple

import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
import pyrenew.observations.infection_functions as inf
from numpy.typing import ArrayLike
from pyrenew.distutil import (
    reverse_discrete_dist_vector,
    validate_discrete_dist_vector,
)
from pyrenew.metaclasses import RandomProcess

InfectionsSample = namedtuple(
    "InfectionsSample",
    ["predicted", "observed"],
    defaults=[None, None],
)
"""Output from InfectionsObservation.sample()"""


class InfectionsObservation(RandomProcess):
    def __init__(
        self,
        gen_int: ArrayLike,
        I0_varname: str = "I0",
        Rt_varname: str = "Rt",
        infections_mean_varname: str = "infections_mean",
        infections_obs_varname: str = "infections_obs",
        I0_dist: dist.Distribution = dist.LogNormal(2, 0.25),
        inf_observation_model: RandomProcess = None,
    ) -> None:
        """Observation of Infections given Rt (Random Process)

        Parameters
        ----------
        gen_int: ArrayLike
            A vector representing the pmf of the generation interval
        I0_varname : str.
            Name of the element in `random_variables` that will hold the value
            of 'I0'.
        Rt_varname : str.
            Name of the element in `random_variables` that will hold the value
            of 'Rt'.
        infections_mean_varname : str.
            Name of the element in `random_variables` that will hold the value
            of mean 'infections'.
        infections_obs_varname : str.
            Name of the element in `random_variables` that will hold the value
            of observed 'infections'.
        I0_dist : dist.Distribution, optional
            Distribution from where to sample the baseline number of infections.

        inf_observation_model : RandomProcess, optional
            RandomProcess representing an observation process in which the
            deterministic number of infections is used as a parameter, defaults
            to None.

        Returns
        -------
        RandomProcess
        """
        InfectionsObservation.validate(I0_dist, gen_int)

        self.I0_dist = I0_dist
        self.gen_int_rev = reverse_discrete_dist_vector(gen_int)

        if inf_observation_model is not None:
            self.obs_model = lambda random_variables, constants: inf_observation_model.sample(
                random_variables=random_variables,
                constants=constants,
            )
        else:
            self.obs_model = lambda random_variables, constants: (
                random_variables.get(self.infections_obs_varname),
            )

        self.I0_varname = I0_varname
        self.Rt_varname = Rt_varname
        self.infections_mean_varname = infections_mean_varname
        self.infections_obs_varname = infections_obs_varname

        return None

    @staticmethod
    def validate(I0_dist, gen_int) -> None:
        assert isinstance(I0_dist, dist.Distribution)
        validate_discrete_dist_vector(gen_int)

        return None

    def sample(
        self,
        random_variables: dict,
        constants: dict = None,
    ) -> InfectionsSample:
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
        """
        I0 = npro.sample(
            name="I0",
            fn=self.I0_dist,
            obs=random_variables.get(self.I0_varname, None),
        )

        n_lead = self.gen_int_rev.size - 1
        I0_vec = jnp.hstack([jnp.zeros(n_lead), I0])

        all_infections = inf.sample_infections_rt(
            I0=I0_vec,
            Rt=random_variables.get(self.Rt_varname),
            reversed_generation_interval_pmf=self.gen_int_rev,
        )

        npro.deterministic(self.infections_mean_varname, all_infections)

        # If specified, building the rv
        rvars = dict()
        rvars[self.infections_mean_varname] = all_infections
        rvars[self.infections_obs_varname] = random_variables.get(
            self.infections_obs_varname, None
        )

        observed, *_ = self.obs_model(
            random_variables=rvars,
            constants=constants,
        )

        return InfectionsSample(all_infections, observed)
