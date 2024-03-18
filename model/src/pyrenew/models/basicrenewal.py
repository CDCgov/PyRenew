#!/usr/bin/env/python
# -*- coding: utf-8 -*-

from collections import namedtuple

from pyrenew.metaclasses import Model, RandomProcess
from pyrenew.processes import RtRandomWalkProcess

# Output class of the BasicRenewalModel
BasicModelSample = namedtuple(
    "InfectModelSample",
    ["Rt", "infect_predicted", "infect_observed"],
)
"""Output from BasicRenewalModel.model()"""


class BasicRenewalModel(Model):
    """
    Implementation of a basic
    renewal model.
    """

    def __init__(
        self,
        infections_obs: RandomProcess,
        Rt_process: RandomProcess = RtRandomWalkProcess(),
    ) -> None:
        """_summary_

        Args:
            infections_obs (RandomProcess): Observation process of infections.
            Rt_process (RandomProcess, optional): _description_. Defaults to
            RtRandomWalkProcess().
        """
        self.infections_obs = infections_obs
        self.Rt_process = Rt_process

    @staticmethod
    def validate(infections_obs, hospitalizations_obs) -> None:
        return None

    def sample_rt(self, random_variables: dict = None, constants: dict = None):
        return self.Rt_process.sample(
            random_variables=random_variables,
            constants=constants,
        )

    def sample_infections(
        self, random_variables: dict = None, constants: dict = None
    ):
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

        :param random_variables: A dictionary containing `infections` and/or
            `Rt` (optional).
        :type random_variables: dict, optional
        :param constants: A dictionary containing `n_timepoints`.
        :type constants: dict, optional
        :return: _description_
        :rtype: _type_
        """

        Rt = self.sample_rt(
            random_variables=random_variables,
            constants=constants,
        )

        if random_variables is None:
            random_variables = dict()

        infect_predicted, infect_observed = self.sample_infections(
            random_variables={**random_variables, **dict(Rt=Rt)},
            constants=constants,
        )

        return BasicModelSample(
            Rt=Rt,
            infect_predicted=infect_predicted,
            infect_observed=infect_observed,
        )
