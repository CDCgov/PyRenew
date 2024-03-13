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
    ):
        self.validate(I0_dist, gen_int)

        self.I0_dist = I0_dist
        self.gen_int_rev = reverse_discrete_dist_vector(gen_int)

        return None

    @staticmethod
    def validate(I0_dist, gen_int) -> None:
        assert isinstance(I0_dist, dist.Distribution)
        validate_discrete_dist_vector(gen_int)

        return None

    def sample(self, data, Rt):
        I0 = npro.sample("I0", self.I0_dist, obs=data.get("I0", None))

        n_lead = self.gen_int_rev.size - 1
        I0_vec = jnp.hstack([jnp.zeros(n_lead), I0])

        all_infections = inf.sample_infections_rt(I0_vec, Rt, self.gen_int_rev)
        npro.deterministic("incidence", all_infections)
        return all_infections
