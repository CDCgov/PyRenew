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


class InfectionsObservation(RandomProcess):
    def __init__(
        self,
        gen_int: ArrayLike,
        I0_dist: dist.Distribution = dist.LogNormal(2, 0.25),
        inf_observation_model: dist.Distribution = None,
    ):
        """
        Observation of Infections given Rt (Random Process)

        :param gen_int: A vector representing the pmf of the generation interval
        :type gen_int: ArrayLike
        :param I0_dist: Distribution from where to sample the baseline number of
            infections, defaults to dist.LogNormal(2, 0.25)
        :type I0_dist: dist.Distribution, optional
        :param inf_observation_model:  Distribution representing an observation
            process in which the deterministic number of infections is used as a
            parameter, defaults to None.
        :type inf_observation_model: dist.Distribution, optional
        :return: A RandomProcess
        :rtype: pyrenew.metaclasses.RandomProcess()
        """
        self.validate(I0_dist, gen_int)

        self.I0_dist = I0_dist
        self.gen_int_rev = reverse_discrete_dist_vector(gen_int)

        if inf_observation_model is not None:
            self.obs_model = lambda obs, data: inf_observation_model.sample(
                obs=obs,
                data=data,
            )
        else:
            self.obs_model = lambda obs, data: obs.get(
                "counts", obs.get("rate")
            )

        return None

    @staticmethod
    def validate(I0_dist, gen_int) -> None:
        assert isinstance(I0_dist, dist.Distribution)
        validate_discrete_dist_vector(gen_int)

        return None

    def sample(
        self,
        obs: dict,
        data: dict = dict(),
    ):
        """Samples infections given Rt

        :param obs: A dictionary containing an observed `Rt` sequence passed to
            `sample_infections_rt()`. It can also contain `infections` and `I0`,
            both passed to `obs` in `numpyro.sample()`.
        :type obs: dict, optional
        :param data: Ignored
        :type data: dict
        :return: _description_
        :rtype: _type_
        """
        I0 = npro.sample("I0", self.I0_dist, obs=obs.get("I0", None))

        n_lead = self.gen_int_rev.size - 1
        I0_vec = jnp.hstack([jnp.zeros(n_lead), I0])

        all_infections = inf.sample_infections_rt(
            I0=I0_vec,
            Rt=obs.get("Rt"),
            reversed_generation_interval_pmf=self.gen_int_rev,
        )

        npro.deterministic("incidence", all_infections)

        observed = self.obs_model(
            obs=dict(
                rate=all_infections,
                counts=obs.get("infections", None),
            ),
            data=data,
        )

        return observed
