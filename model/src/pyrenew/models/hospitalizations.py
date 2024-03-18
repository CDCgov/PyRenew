#!/usr/bin/env/python
# -*- coding: utf-8 -*-

from collections import namedtuple

from pyrenew.metaclasses import RandomProcess, _assert_sample_and_rtype
from pyrenew.models.basicrenewal import BasicRenewalModel
from pyrenew.processes import RtRandomWalkProcess

HospModelSample = namedtuple(
    "HospModelSample",
    ["Rt", "infect_sampled", "IHR", "pred_hosps", "samp_hosp"],
)
"""Output from HospitalizationsModel.model()
"""


class HospitalizationsModel(BasicRenewalModel):
    """Hospitalizations Model (BasicRenewal + Hospitalizations)

    This class inherits from pyrenew.models.BasicRenewalModel. It extends the
    basic renewal model by adding a hospitalization module, e.g.,
    pyrenew.observations.HospitalizationsObservation.
    """

    def __init__(
        self,
        infections_obs: RandomProcess,
        hosp_obs: RandomProcess,
        Rt_process: RandomProcess = RtRandomWalkProcess(),
    ) -> None:
        """Default constructor

        :param infections_obs: The infections observation process.
        :type infections_obs: RandomProcess
        :param hosp_obs: _description_
        :type hosp_obs: RandomProcess
        :param Rt_process: _description_, defaults to RtRandomWalkProcess()
        :type Rt_process: RandomProcess, optional
        """
        self.validate(infections_obs, hosp_obs, Rt_process)

        BasicRenewalModel.__init__(
            self,
            infections_obs=infections_obs,
            Rt_process=Rt_process,
        )

        self.hosp_obs = hosp_obs

    @staticmethod
    def validate(infections_obs, hosp_obs, Rt_process) -> None:
        _assert_sample_and_rtype(infections_obs)
        _assert_sample_and_rtype(hosp_obs)
        _assert_sample_and_rtype(Rt_process)
        return None

    def sample_hospitalizations(
        self,
        random_variables: dict,
        constants: dict = None,
    ) -> tuple:
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

    def model(
        self,
        random_variables: dict = None,
        constants: dict = None,
    ) -> HospModelSample:
        """Hospitalizations model

        :param random_variables: A dictionary with random variables passed to
            `pyrenew.models.BasicRenewalModel` and `sample_hospitalizations`;
            defaults to None.
        :type random_variables: dict, optional
        :param constants: _description_, defaults to None
        :type constants: dict, optional
        :return: _description_
        :rtype: HospModelSample
        """
        if random_variables is None:
            random_variables = dict()

        if constants is None:
            constants = dict()

        Rt, infect_sampled, *_ = BasicRenewalModel.model(
            self, constants=constants
        )

        IHR, pred_hosps, samp_hosp, *_ = self.sample_hospitalizations(
            random_variables={
                **random_variables,
                **dict(infections=infect_sampled),
            },
            constants=constants,
        )

        return HospModelSample(
            Rt=Rt,
            infect_sampled=infect_sampled,
            IHR=IHR,
            pred_hosps=pred_hosps,
            samp_hosp=samp_hosp,
        )
