#!/usr/bin/env/python
# -*- coding: utf-8 -*-

from pyrenew.metaclasses import Model, RandomProcess
from pyrenew.processes import RtRandomWalkProcess


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
        self.observed_infections = None

    @staticmethod
    def validate(infections_obs, hospitalizations_obs) -> None:
        return None

    def sample_rt(self, data):
        return self.Rt_process.sample(data)

    def sample_infections(self, data, Rt, obs=None):
        return self.infections_obs.sample(data, Rt, obs=obs)

    def set_observed_infections(self, obs):
        self.observed_infections = obs

    def model(self, data=None):
        if data is None:
            data = dict()

        Rt = self.sample_rt(data=data)

        infections = self.sample_infections(
            data=data, Rt=Rt, obs=self.observed_infections
        )

        return Rt, infections
