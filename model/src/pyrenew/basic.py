#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
import pyrenew.infection as inf
from pyrenew.distutil import (
    reverse_discrete_dist_vector,
    validate_discrete_dist_vector,
)
from pyrenew.observation import PoissonObservation
from pyrenew.process import SimpleRandomWalk
from pyrenew.transform import LogTransform


class BasicRenewalModel:
    """
    Implementation of a basic
    renewal model, not abstracted
    or modular, just for testing
    """

    def __init__(
        self,
        Rt0_dist=None,
        Rt_transform=None,
        Rt_rw_dist=None,
        I0_dist=None,
        IHR_dist=None,
        gen_int=None,
        inf_hosp_int=None,
        hosp_observation_model=None,
    ):
        if Rt_transform is None:
            Rt_transform = LogTransform()
        self.Rt_transform = Rt_transform

        if Rt0_dist is None:
            Rt0_dist = dist.TruncatedNormal(loc=1.2, scale=0.2, low=0)
        self.Rt0_dist = Rt0_dist

        if Rt_rw_dist is None:
            Rt_rw_dist = dist.Normal(0, 0.025)
        self.Rt_rw_dist = Rt_rw_dist

        if I0_dist is None:
            I0_dist = dist.LogNormal(2, 0.25)
        self.I0_dist = I0_dist

        if IHR_dist is None:
            IHR_dist = dist.LogNormal(jnp.log(0.05), 0.05)
        self.IHR_dist = IHR_dist

        self.gen_int_rev = reverse_discrete_dist_vector(
            validate_discrete_dist_vector(gen_int)
        )
        self.inf_hosp = validate_discrete_dist_vector(inf_hosp_int)

        if hosp_observation_model is None:
            hosp_observation_model = PoissonObservation()
        self.hosp_observation_model = hosp_observation_model

    def sample_rt(self, data):
        n_timepoints = data["n_timepoints"]

        Rt0 = npro.sample("Rt0", self.Rt0_dist, obs=data.get("Rt0", None))

        Rt0_trans = self.Rt_transform(Rt0)
        Rt_trans_proc = SimpleRandomWalk(self.Rt_rw_dist)
        Rt_trans_ts = Rt_trans_proc.sample(
            duration=n_timepoints, name="Rt_transformed_rw", init=Rt0_trans
        )

        Rt = npro.deterministic("Rt", self.Rt_transform.inverse(Rt_trans_ts))

        return Rt

    def sample_infections(self, data, Rt):
        I0 = npro.sample("I0", self.I0_dist, obs=data.get("I0", None))

        n_lead = self.gen_int_rev.size - 1
        I0_vec = jnp.hstack([jnp.zeros(n_lead), I0])

        all_infections = inf.sample_infections_rt(I0_vec, Rt, self.gen_int_rev)
        npro.deterministic("incidence", all_infections)

        return all_infections

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
