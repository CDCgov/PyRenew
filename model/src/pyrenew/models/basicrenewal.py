#!/usr/bin/env/python
# -*- coding: utf-8 -*-

from collections import namedtuple

from pyrenew.metaclasses import Model, RandomProcess, _assert_sample_and_rtype
from pyrenew.processes import RtRandomWalkProcess

# Output class of the BasicRenewalModel
BasicModelSample = namedtuple(
    "InfectModelSample",
    ["Rt", "infect_predicted", "infect_observed"],
    defaults=[None, None, None],
)
"""Output from BasicRenewalModel.model()"""


class BasicRenewalModel(Model):
    """Basic Renewal Model (Infections + Rt)

    The basic renewal model consists of a sampler of two steps: Sample from
    Rt and then used that to sample the infections.
    """

    def __init__(
        self,
        infections_obs: RandomProcess,
        Rt_process: RandomProcess = RtRandomWalkProcess(),
    ) -> None:
        """Default constructor

        Parameters
        ----------
        infections_obs : RandomProcess
            Infections observation process (e.g.,
            pyrenew.observations.InfectionsObservation.)
        Rt_process : RandomProcess, optional
            The sample function of the process should return a tuple where the
            first element is the drawn Rt., by default RtRandomWalkProcess()

        Returns
        -------
        None
        """

        BasicRenewalModel.validate(
            infections_obs=infections_obs,
            Rt_process=Rt_process,
        )

        self.infections_obs = infections_obs
        self.Rt_process = Rt_process

    @staticmethod
    def validate(infections_obs, Rt_process) -> None:
        _assert_sample_and_rtype(infections_obs, skip_if_none=False)
        _assert_sample_and_rtype(Rt_process, skip_if_none=False)
        return None

    def sample_rt(
        self,
        random_variables: dict = None,
        constants: dict = None,
    ) -> tuple:
        return self.Rt_process.sample(
            random_variables=random_variables,
            constants=constants,
        )

    def sample_infections(
        self,
        random_variables: dict = None,
        constants: dict = None,
    ) -> tuple:
        return self.infections_obs.sample(
            random_variables=random_variables,
            constants=constants,
        )

    def model(
        self,
        random_variables: dict = None,
        constants: dict = None,
    ) -> BasicModelSample:
        """_summary_

        Parameters
        ----------
        random_variables : dict, optional
            A dictionary containing `infections` and/or `Rt` (optional).
        constants : dict, optional
            A dictionary containing `n_timepoints`.

        Returns
        -------
        BasicModelSample
        """

        if random_variables is None:
            random_variables = dict()

        if constants is None:
            constants = dict()

        # Sampling from Rt (possibly with a given Rt, depending on
        # the Rt_process (RandomProcess) object.)
        Rt, *_ = self.sample_rt(
            random_variables=random_variables,
            constants=constants,
        )

        # Sampling infections. Possibly, if infections are passed via
        # `random_variables`, the model will pass that to numpyro.sample.
        infect_predicted, infect_observed, *_ = self.sample_infections(
            random_variables={**random_variables, **dict(Rt=Rt)},
            constants=constants,
        )

        return BasicModelSample(
            Rt=Rt,
            infect_predicted=infect_predicted,
            infect_observed=infect_observed,
        )
