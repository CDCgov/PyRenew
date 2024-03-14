#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
from numpy.typing import ArrayLike
from pyrenew.distutil import validate_discrete_dist_vector
from pyrenew.metaclasses import RandomProcess


class HospitalizationsObservation(RandomProcess):
    """Observed hospitalizations random process"""

    def __init__(
        self,
        inf_hosp_int: ArrayLike,
        IHR_dist: dist.Distribution = dist.LogNormal(jnp.log(0.05), 0.05),
    ) -> None:
        """Default constructor

        :param inf_hosp_int: pmf for reporting (informing) hospitalizations.
        :type inf_hosp_int: ArrayLike
        :param IHR_dist: Infection to hospitalization rate pmf, defaults to
            dist.LogNormal(jnp.log(0.05), 0.05)
        :type IHR_dist: dist.Distribution, optional
        """
        self.validate(IHR_dist)

        self.IHR_dist = IHR_dist
        self.inf_hosp = validate_discrete_dist_vector(inf_hosp_int)

    @staticmethod
    def validate(IHR_dist) -> None:
        assert isinstance(IHR_dist, dist.Distribution)

        return None

    def sample(self, infections: ArrayLike, data: dict = dict()):
        """Samples from the observation process

        :param infections: Vector of infections (could be latent variable).
        :type infections: ArrayLike
        :param data: A dictionary with possible a vector of IHR, defaults to
            None
        :type data: dict, optional
        :return: _description_
        :rtype: _type_
        """
        IHR = npro.sample("IHR", self.IHR_dist, obs=data.get("IHR", None))

        IHR_t = IHR * infections

        pred_hosps = jnp.convolve(IHR_t, self.inf_hosp, mode="full")[
            : IHR_t.shape[0]
        ]

        npro.deterministic("predicted_hospital_admissions", pred_hosps)

        return IHR, pred_hosps
