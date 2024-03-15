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

    @staticmethod
    def validate(infections_obs, hospitalizations_obs) -> None:
        return None

    def sample_rt(self, obs=dict(), data=dict()):
        return self.Rt_process.sample(obs=obs, data=data)

    def sample_infections(self, obs=dict(), data=dict()):
        return self.infections_obs.sample(obs=obs, data=data)

    def model(self, obs=dict(), data=dict()):
        Rt = self.sample_rt(obs=obs, data=data)

        infections = self.sample_infections(
            obs={**obs, **dict(Rt=Rt)},
            data=data,
        )

        return Rt, infections
