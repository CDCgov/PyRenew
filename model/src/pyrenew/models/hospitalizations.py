#!/usr/bin/env/python
# -*- coding: utf-8 -*-

from pyrenew.models.basicrenewal import BasicRenewalModel
from pyrenew.observations import HospitalizationsObservation
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
        inf_hosp_ini,  # inf_hosp_int: pmf for reporting (informing) hospitalizations.
        Rt_process=RtRandomWalkProcess(),
        hosp_dist=None,
        IHR_dist=None,
    ) -> None:
        BasicRenewalModel.__init__(self, infections_obs, Rt_process)

        self.hosp_sampler = HospitalizationsObservation(
            inf_hosp_int=inf_hosp_ini,
            hosp_dist=hosp_dist,
            IHR_dist=None,
        )

    @staticmethod
    def validate() -> None:
        return None

    def sample_hospitalizations(self, obs: dict, data: dict = dict()):
        """Sample number of hospitalizations

        :param obs: A dictionary containing `infections` passed to the specified
            sampler.
        :type obs: dict
        :param data: _description_, defaults to dict()
        :type data: dict, optional
        :return: _description_
        :rtype: _type_
        """

        return self.hosp_sampler.sample(obs=obs, data=data)

    def observe_hospitalizations(
        self,
        infections=None,
        IHR=None,
        pred_hosps=None,
        obs=dict(),
        data=dict(),
    ):
        return self.hosp_observation_model.sample(
            predicted_value=pred_hosps,
            obs=dict(obs.get("observed_hospitalizations", None)),
            data=data,
        )

    def model(self, obs=dict(), data=dict()):
        Rt, infections = BasicRenewalModel.model(self, data=data)

        IHR, pred_hosps, samp_hosp = self.sample_hospitalizations(
            obs=obs, data={**data, **dict(infections=infections)}
        )

        return Rt, infections, IHR, pred_hosps, samp_hosp
