#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
from pyrenew.distutil import validate_discrete_dist_vector
from pyrenew.metaclasses import Model
from pyrenew.observations import PoissonObservation
from pyrenew.processes import RtRandomWalkProcess


class BasicRenewalModel(Model):
    """
    Implementation of a basic
    renewal model, not abstracted
    or modular, just for testing
    """

    def __init__(
        self,
        infections_obs,
        Rt_process=RtRandomWalkProcess(),
        IHR_dist=None,
        inf_hosp_int=None,
        hosp_observation_model=None,
    ):
        self.Rt_process = Rt_process
        self.infections_obs = infections_obs

        if IHR_dist is None:
            IHR_dist = dist.LogNormal(jnp.log(0.05), 0.05)
        self.IHR_dist = IHR_dist
        self.inf_hosp = validate_discrete_dist_vector(inf_hosp_int)

        if hosp_observation_model is None:
            hosp_observation_model = PoissonObservation()
        self.hosp_observation_model = hosp_observation_model

    def sample_rt(self, data):
        return self.Rt_process.sample(data)

    def sample_infections(self, data, Rt):
        return self.infections_obs.sample(data, Rt)

    def sample_hospitalizations(self, data=None, Rt=None, infections=None):
        IHR = npro.sample("IHR", self.IHR_dist, obs=data.get("IHR", None))

        IHR_t = IHR * infections

        pred_hosps = jnp.convolve(IHR_t, self.inf_hosp, mode="full")[
            : IHR_t.shape[0]
        ]

        npro.deterministic("predicted_hospital_admissions", pred_hosps)

        return IHR, pred_hosps

    def observe_hospitalizations(
        self, data=None, Rt=None, infections=None, IHR=None, pred_hosps=None
    ):
        return self.hosp_observation_model.sample(
            parameter_name="hospitalizations",
            predicted_value=pred_hosps,
            data=data,
            obs=data.get("observed_hospitalizations", None),
        )

    def model(self, data=None):
        if data is None:
            data = dict()

        Rt = self.sample_rt(data=data)

        infections = self.sample_infections(data=data, Rt=Rt)

        IHR, pred_hosps = self.sample_hospitalizations(
            data=data, Rt=Rt, infections=infections
        )

        obs_hosps = self.observe_hospitalizations(
            data=data,
            Rt=Rt,
            infections=infections,
            IHR=IHR,
            pred_hosps=pred_hosps + 1e-20,
        )

        return Rt, infections, IHR, pred_hosps, obs_hosps
