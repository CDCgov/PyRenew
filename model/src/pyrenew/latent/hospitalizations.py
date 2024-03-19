#!/usr/bin/env/python
# -*- coding: utf-8 -*-

from collections import namedtuple

import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
from numpy.typing import ArrayLike
from pyrenew.distutil import validate_discrete_dist_vector
from pyrenew.metaclasses import RandomProcess

HospSampledObs = namedtuple(
    "HospSampledObs",
    ["IHR", "predicted"],
    defaults=[None, None],
)
"""Output from Hospitalizations.sample()"""


class Hospitalizations(RandomProcess):
    """Observed hospitalizations random process"""

    def __init__(
        self,
        inf_hosp_int: ArrayLike,
        infections_varname: str = "infections",
        IHR_obs_varname: str = "IHR",
        hospitalizations_predicted_varname: str = "hospitalizations_predicted",
        IHR_dist: dist.Distribution = dist.LogNormal(jnp.log(0.05), 0.05),
    ) -> None:
        """Default constructor

        Parameters
        ----------
        inf_hosp_int : ArrayLike
            pmf for reporting (informing) hospitalizations.
        infections_varname : str
            Name of the entry in random_variables that holds the vector of
            infections.
        IHR_obs_varname : str
            Name of the entry in random_variables that holds the observed IHR
            (if available).
        hospitalizations_predicted_varname : str
            Name to assign to the deterministic component in numpyro of
            predicted hospitalizations.
        IHR_dist : dist.Distribution, optional
            Infection to hospitalization rate pmf.

        Returns
        -------
        None
        """
        Hospitalizations.validate(IHR_dist)

        self.IHR_obs_varname = IHR_obs_varname
        self.infections_varname = infections_varname
        self.hospitalizations_predicted_varname = (
            hospitalizations_predicted_varname
        )

        self.IHR_dist = IHR_dist
        self.inf_hosp = validate_discrete_dist_vector(inf_hosp_int)

    @staticmethod
    def validate(IHR_dist) -> None:
        assert isinstance(IHR_dist, dist.Distribution)

        return None

    def sample(
        self,
        random_variables: dict = None,
        constants: dict = None,
    ) -> HospSampledObs:
        """Samples from the observation process
        :param random_variables: A dictionary with `IHR` passed to `obs` in
            `npyro.sample()`.
        :type random_variables: dict
        :param constants: A dictionary with observed `infections`.
        :type constants: dict, optional
        :return: _description_
        :rtype: _type_
        """

        if random_variables is None:
            random_variables = dict()

        if constants is None:
            constants = dict()

        IHR = npro.sample(
            "IHR",
            self.IHR_dist,
            obs=random_variables.get(self.IHR_obs_varname, None),
        )

        IHR_t = IHR * random_variables.get(self.infections_varname)

        pred_hosps = jnp.convolve(IHR_t, self.inf_hosp, mode="full")[
            : IHR_t.shape[0]
        ]

        npro.deterministic(self.hospitalizations_predicted_varname, pred_hosps)

        return HospSampledObs(IHR, pred_hosps)
