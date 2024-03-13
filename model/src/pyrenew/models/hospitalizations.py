#!/usr/bin/env/python
# -*- coding: utf-8 -*-

from pyrenew.metaclasses import RandomProcess
from pyrenew.models.basicrenewal import BasicRenewalModel
from pyrenew.observations import PoissonObservation
from pyrenew.processes import RtRandomWalkProcess


class HospitalizationsModel(BasicRenewalModel):
    """
    Implementation of a basic
    renewal model, not abstracted
    or modular, just for testing
    """

    def __init__(
        self,
        infections_obs,
        hospitalizations_obs,
        Rt_process=RtRandomWalkProcess(),
        hosp_observation_model=PoissonObservation(),
    ) -> None:
        BasicRenewalModel.__init__(self, infections_obs, Rt_process)

        self.validate(hospitalizations_obs, hosp_observation_model)

        self.hospitalizations_obs = hospitalizations_obs
        self.hosp_observation_model = hosp_observation_model

    @staticmethod
    def validate(hospitalizations_obs, hosp_observation_model) -> None:
        assert isinstance(hospitalizations_obs, RandomProcess)
        assert isinstance(hosp_observation_model, RandomProcess)
        return None

    def sample_hospitalizations(self, data=None, Rt=None, infections=None):
        return self.hospitalizations_obs.sample(
            data=data,
            Rt=Rt,
            infections=infections,
        )

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

        Rt, infections = BasicRenewalModel.model(self, data=data)

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
