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
            self.obs_model = (
                lambda predicted_value, obs: inf_observation_model.sample(
                    predicted_value=predicted_value,
                    obs=obs,
                )
            )
        else:
            self.obs_model = lambda predicted_value, obs: predicted_value

        return None

    @staticmethod
    def validate(I0_dist, gen_int) -> None:
        assert isinstance(I0_dist, dist.Distribution)
        validate_discrete_dist_vector(gen_int)

        return None

    def sample(self, Rt, data: dict = dict(), obs=None):
        """Samples infections given Rt

        :param data: A dictionary possibly containing `I0`. Defaults to dict().
        :type data: _type_
        :param Rt: A vector containing an observed `Rt` sequence.
        :type Rt: _type_
        :param obs: Observed vector of infections, defaults to None
        :type obs: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        I0 = npro.sample("I0", self.I0_dist, obs=data.get("I0", None))

        n_lead = self.gen_int_rev.size - 1
        I0_vec = jnp.hstack([jnp.zeros(n_lead), I0])

        all_infections = inf.sample_infections_rt(I0_vec, Rt, self.gen_int_rev)
        npro.deterministic("incidence", all_infections)

        observed = self.obs_model(
            all_infections,
            obs=obs,
        )

        return observed
