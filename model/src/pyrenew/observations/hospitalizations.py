#!/usr/bin/env/python
# -*- coding: utf-8 -*-

from collections import namedtuple

import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
from numpy.typing import ArrayLike
from pyrenew.distutil import validate_discrete_dist_vector
from pyrenew.metaclasses import RandomProcess, _assert_sample_and_rtype

HospSampledObs = namedtuple(
    "HospSampledObs",
    ["IHR", "predicted", "sampled"],
)
"""Output from HospitalizationsObservation.sample()"""


class HospitalizationsObservation(RandomProcess):
    """Observed hospitalizations random process"""

    def __init__(
        self,
        inf_hosp_int: ArrayLike,
        IHR_obs_varname: str = "IHR_obs",
        infections_obs_varname: str = "infections_obs",
        hospitalizations_predicted_varname: str = "hospitalizations_predicted",
        hospitalizations_obs_varname: str = "hospitalizations_obs",
        hosp_dist: RandomProcess = None,
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

        self.hosp_dist = hosp_dist
        self.IHR_obs_varname = IHR_obs_varname
        self.infections_obs_varname = infections_obs_varname
        self.hospitalizations_predicted_varname = (
            hospitalizations_predicted_varname
        )
        self.hospitalizations_obs_varname = hospitalizations_obs_varname

        if hosp_dist is not None:
            self.sample_hosp = (
                lambda random_variables, constants: self.hosp_dist.sample(
                    random_variables=random_variables, constants=constants
                )
            )
        else:
            self.sample_hosp = lambda random_variables, constants: (None,)

        self.IHR_dist = IHR_dist
        self.inf_hosp = validate_discrete_dist_vector(inf_hosp_int)

    @staticmethod
    def validate(hosp_dist, IHR_dist) -> None:
        _assert_sample_and_rtype(hosp_dist)
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

        IHR_t = IHR * random_variables.get(self.infections_obs_varname)

        pred_hosps = jnp.convolve(IHR_t, self.inf_hosp, mode="full")[
            : IHR_t.shape[0]
        ]

        npro.deterministic(self.hospitalizations_predicted_varname, pred_hosps)

        # Preparing dict
        rvars = dict()
        rvars[self.hospitalizations_predicted_varname] = pred_hosps
        rvars[self.hospitalizations_obs_varname] = random_variables.get(
            self.hospitalizations_obs_varname, None
        )

        sampled_hosps, *_ = self.sample_hosp(
            random_variables=rvars,
            constants=constants,
        )

        return HospSampledObs(IHR, pred_hosps, sampled_hosps)
