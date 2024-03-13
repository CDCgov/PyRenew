#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
from pyrenew.distutil import validate_discrete_dist_vector
from pyrenew.metaclasses import RandomProcess


class HospitalizationsObservation(RandomProcess):
    def __init__(
        self,
        inf_hosp_int,
        IHR_dist: dist.Distribution = dist.LogNormal(jnp.log(0.05), 0.05),
    ) -> None:
        self.validate(IHR_dist)

        self.IHR_dist = IHR_dist
        self.inf_hosp = validate_discrete_dist_vector(inf_hosp_int)

    @staticmethod
    def validate(IHR_dist) -> None:
        assert isinstance(IHR_dist, dist.Distribution)

        return None

    def sample(self, data=None, Rt=None, infections=None):
        IHR = npro.sample("IHR", self.IHR_dist, obs=data.get("IHR", None))

        IHR_t = IHR * infections

        pred_hosps = jnp.convolve(IHR_t, self.inf_hosp, mode="full")[
            : IHR_t.shape[0]
        ]

        npro.deterministic("predicted_hospital_admissions", pred_hosps)

        return IHR, pred_hosps
