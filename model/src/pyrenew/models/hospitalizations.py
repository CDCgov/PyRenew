#!/usr/bin/env/python
# -*- coding: utf-8 -*-

from collections import namedtuple

from pyrenew.metaclasses import Model, RandomProcess, _assert_sample_and_rtype
from pyrenew.models.basicrenewal import BasicRenewalModel
from pyrenew.processes import RtRandomWalkProcess

HospModelSample = namedtuple(
    "HospModelSample",
    ["Rt", "infect_sampled", "IHR", "pred_hosps", "samp_hosp"],
    defaults=[None, None, None, None, None],
)
"""Output from HospitalizationsModel.sample()
"""


class HospitalizationsModel(Model):
    """Hospitalizations Model (BasicRenewal + Hospitalizations)

    This class inherits from pyrenew.models.Model. It extends the
    basic renewal model by adding a hospitalization module, e.g.,
    pyrenew.observations.Hospitalizations.
    """

    def __init__(
        self,
        hosp_latent: RandomProcess,
        hosp_obs: RandomProcess,
        infections_latent: RandomProcess,
        infections_obs: RandomProcess = None,
        Rt_process: RandomProcess = RtRandomWalkProcess(),
    ) -> None:
        """Default constructor

        Parameters
        ----------
        type hosp_obs : RandomProcess
            Observation process for the hospitalizations.
        type infections_obs : RandomProcess
            The infections observation process (passed to BasicRenewalModel).
        Rt_process : RandomProcess, optional
            Rt process  (passed to BasicRenewalModel).

        Returns
        -------
        None
        """
        self.basic_renewal = BasicRenewalModel(
            infections_latent=infections_latent,
            infections_obs=infections_obs,
            Rt_process=Rt_process,
        )

        HospitalizationsModel.validate(hosp_latent, hosp_obs)

        self.hosp_latent = hosp_latent
        self.hosp_obs = hosp_obs

    @staticmethod
    def validate(hosp_latent, hosp_obs) -> None:
        _assert_sample_and_rtype(hosp_latent, skip_if_none=False)
        _assert_sample_and_rtype(hosp_obs, skip_if_none=True)
        return None

    def sample_hospitalizations_latent(
        self,
        random_variables: dict,
        constants: dict = None,
    ) -> tuple:
        return self.hosp_latent.sample(
            random_variables=random_variables,
            constants=constants,
        )

    def sample_hospitalizations_obs(
        self,
        random_variables: dict,
        constants: dict = None,
    ) -> tuple:
        """Sample number of hospitalizations

        Parameters
        ----------
        random_variables : dict
            A dictionary containing `infections` passed to the specified
            sampler.
        constants : dict, optional
            Possible constants for the model.

        Returns
        -------
        tuple
        """

        return self.hosp_obs.sample(
            random_variables=random_variables,
            constants=constants,
        )

    def sample(
        self,
        random_variables: dict = None,
        constants: dict = None,
    ) -> HospModelSample:
        """Sample from the Hospitalizations model

        Parameters
        ----------
        random_variables : dict, optional
            A dictionary with random variables passed to
            `pyrenew.models.BasicRenewalModel` and `sample_hospitalizations`.
        constants : dict, optional
            Possible constants for the model.

        Returns
        -------
        HospModelSample
        """
        if random_variables is None:
            random_variables = dict()

        if constants is None:
            constants = dict()

        Rt, infect_sampled, *_ = self.basic_renewal.sample(constants=constants)

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
