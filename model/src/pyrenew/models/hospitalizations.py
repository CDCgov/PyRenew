#!/usr/bin/env/python
# -*- coding: utf-8 -*-

from collections import namedtuple

from pyrenew.metaclasses import Model, RandomProcess, _assert_sample_and_rtype
from pyrenew.models.basicrenewal import BasicRenewalModel
from pyrenew.processes import RtRandomWalkProcess

HospModelSample = namedtuple(
    "HospModelSample",
    [
        "Rt",
        "infect_sampled",
        "IHR",
        "predicted_hospitalizations",
        "sampled_hosp",
    ],
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
        latent_hospitalizations: RandomProcess,
        observed_hospitalizations: RandomProcess,
        latent_infections: RandomProcess,
        observed_infections: RandomProcess = None,
        Rt_process: RandomProcess = RtRandomWalkProcess(),
    ) -> None:
        """Default constructor

        Parameters
        ----------
        latent_hospitalizations : RandomProcess
            Latent process for the hospitalizations.
        observed_hospitalizations : RandomProcess
            Observation process for the hospitalizations.
        latent_infections : RandomProcess
            The infections latent process (passed to BasicRenewalModel).
        observed_infections : RandomProcess, optional
            The infections observation process (passed to BasicRenewalModel).
        Rt_process : RandomProcess, optional
            Rt process  (passed to BasicRenewalModel).

        Returns
        -------
        None
        """
        self.basic_renewal = BasicRenewalModel(
            latent_infections=latent_infections,
            observed_infections=observed_infections,
            Rt_process=Rt_process,
        )

        HospitalizationsModel.validate(
            latent_hospitalizations, observed_hospitalizations
        )

        self.latent_hospitalizations = latent_hospitalizations
        self.observed_hospitalizations = observed_hospitalizations

    @staticmethod
    def validate(latent_hospitalizations, observed_hospitalizations) -> None:
        _assert_sample_and_rtype(latent_hospitalizations, skip_if_none=False)
        _assert_sample_and_rtype(observed_hospitalizations, skip_if_none=True)
        return None

    def sample_hospitalizations_latent(
        self,
        random_variables: dict,
        constants: dict = None,
    ) -> tuple:
        return self.latent_hospitalizations.sample(
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

        return self.observed_hospitalizations.sample(
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

        Rt, infect_sampled, *_ = self.basic_renewal.sample(
            constants=constants,
            random_variables=random_variables,
        )

        (
            IHR,
            predicted_hospitalizations,
            *_,
        ) = self.sample_hospitalizations_latent(
            random_variables={
                **random_variables,
                **dict(infections=infect_sampled),
            },
            constants=constants,
        )

        sampled_hosp, *_ = self.sample_hospitalizations_obs(
            random_variables={
                **random_variables,
                **dict(predicted_hospitalizations=predicted_hospitalizations),
            },
            constants=constants,
        )

        return HospModelSample(
            Rt=Rt,
            infect_sampled=infect_sampled,
            IHR=IHR,
            predicted_hospitalizations=predicted_hospitalizations,
            sampled_hosp=sampled_hosp,
        )
