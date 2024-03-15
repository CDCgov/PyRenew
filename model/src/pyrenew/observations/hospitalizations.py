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
        hosp_dist: dist.Distribution = None,
        IHR_dist: dist.Distribution = dist.LogNormal(jnp.log(0.05), 0.05),
    ) -> None:
        """Default constructor

        :param inf_hosp_int: pmf for reporting (informing) hospitalizations.
        :type inf_hosp_int: ArrayLike
        :param hosp_dist: If not None, a count distribution receiving a single
            paramater (e.g., `counts` or `rate`.) When specified, the model will
            sample observed hospitalizations from that distribution using the
            predicted hospitalizations as parameter.
        :type hosp_dist: dist.Distribution, optional
        :param IHR_dist: Infection to hospitalization rate pmf, defaults to
            dist.LogNormal(jnp.log(0.05), 0.05)
        :type IHR_dist: dist.Distribution, optional
        """
        self.validate(hosp_dist, IHR_dist)

        if hosp_dist is None:
            self.sample_hosp = lambda obs, data: npro.sample(
                name="sampled_hospitalizations",
                fn=self.hosp_dist(obs.get("predicted_hospitalizations")),
                obs=obs.get("observed_hospitalizations"),
            )
        else:
            self.sample_hosp = lambda obs, data: None

        self.IHR_dist = IHR_dist
        self.inf_hosp = validate_discrete_dist_vector(inf_hosp_int)

    @staticmethod
    def validate(hosp_dist, IHR_dist) -> None:
        if hosp_dist is not None:
            assert isinstance(hosp_dist, dist.Distribution)

        assert isinstance(IHR_dist, dist.Distribution)

        return None

    def sample(
        self,
        obs: dict,
        data: dict,
    ):
        """Samples from the observation process
        :param obs: A dictionary with `IHR` passed to `obs` in `npyro.sample()`.
        :type obs: dict
        :param data: A dictionary with observed `infections`.
        :type data: dict, optional
        :return: _description_
        :rtype: _type_
        """
        IHR = npro.sample("IHR", self.IHR_dist, obs=obs.get("IHR", None))

        IHR_t = IHR * data.get("infections")

        pred_hosps = jnp.convolve(IHR_t, self.inf_hosp, mode="full")[
            : IHR_t.shape[0]
        ]

        npro.deterministic("predicted_hospitalizations", pred_hosps)

        sampled_hosps = self.sample_hosp(
            obs=dict(predicted_hospitalizations=pred_hosps),
            data=data,
        )

        return IHR, pred_hosps, sampled_hosps
