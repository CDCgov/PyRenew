#!/usr/bin/env/python
# -*- coding: utf-8 -*-

from pyrenew.metaclasses import Model
from pyrenew.processes import RtRandomWalkProcess


class BasicRenewalModel(Model):
    """
    Implementation of a basic
    renewal model, not abstracted
    or modular, just for testing
    """

    def __init__(
        self,
        infections_obs,
        Rt_process=RtRandomWalkProcess(),
    ) -> None:
        self.infections_obs = infections_obs
        self.Rt_process = Rt_process

    @staticmethod
    def validate(infections_obs, hospitalizations_obs) -> None:
        return None

    def sample_rt(self, data):
        return self.Rt_process.sample(data)

    def sample_infections(self, data, Rt):
        return self.infections_obs.sample(data, Rt)

    def model(self, data=None):
        if data is None:
            data = dict()

        Rt = self.sample_rt(data=data)

        infections = self.sample_infections(data=data, Rt=Rt)

        return Rt, infections
