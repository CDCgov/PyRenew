#!/usr/bin/env/python
# -*- coding: utf-8 -*-

from collections import namedtuple

from pyrenew.models.basicrenewal import BasicRenewalModel
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
        hosp_obs,
        Rt_process=RtRandomWalkProcess(),
    ) -> None:
        BasicRenewalModel.__init__(
            self,
            infections_obs=infections_obs,
            Rt_process=Rt_process,
        )

        self.hosp_obs = hosp_obs

    @staticmethod
    def validate() -> None:
        return None

    def sample_hospitalizations(
        self,
        random_variables: dict,
        constants: dict = None,
    ):
        """Sample number of hospitalizations

        :param random_variables: A dictionary containing `infections` passed to
            the specified sampler.
        :type random_variables: dict
        :param constants: _description_, defaults to None.
        :type constants: dict, optional
        :return: _description_
        :rtype: _type_
        """

        return self.hosp_obs.sample(
            random_variables=random_variables,
            constants=constants,
        )

    def model(self, random_variables: dict = None, constants: dict = None):
        if random_variables is None:
            random_variables = dict()

        if constants is None:
            constants = dict()

        Rt, infect_sampled, _ = BasicRenewalModel.model(
            self, constants=constants
        )

        IHR, pred_hosps, samp_hosp = self.sample_hospitalizations(
            random_variables=random_variables,
            constants={**constants, **dict(infections=infect_sampled)},
        )

        HospSample = namedtuple(
            "HospSample",
            ["Rt", "infect_sampled", "IHR", "pred_hosps", "samp_hosp"],
        )

        return HospSample(
            Rt=Rt,
            infect_sampled=infect_sampled,
            IHR=IHR,
            pred_hosps=pred_hosps,
            samp_hosp=samp_hosp,
        )
