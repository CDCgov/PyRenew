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

        self.hosp_dist = hosp_dist

        if hosp_dist is not None:
            self.sample_hosp = lambda random_variables, constants: npro.sample(
                name="sampled_hospitalizations",
                fn=self.hosp_dist(
                    random_variables.get("predicted_hospitalizations")
                ),
                obs=random_variables.get("observed_hospitalizations"),
            )
        else:
            self.sample_hosp = lambda random_variables, constants: None

        self.IHR_dist = IHR_dist
        self.inf_hosp = validate_discrete_dist_vector(inf_hosp_int)

    @staticmethod
    def validate(hosp_dist, IHR_dist) -> None:
        # if hosp_dist is not None:
        #     assert isinstance(hosp_dist, dist.Distribution)

        assert isinstance(IHR_dist, dist.Distribution)

        return None

    def sample(
        self,
        random_variables: dict = None,
        constants: dict = None,
    ):
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
            "IHR", self.IHR_dist, obs=random_variables.get("IHR", None)
        )

        IHR_t = IHR * constants.get("infections")

        pred_hosps = jnp.convolve(IHR_t, self.inf_hosp, mode="full")[
            : IHR_t.shape[0]
        ]

        npro.deterministic("predicted_hospitalizations", pred_hosps)

        sampled_hosps = self.sample_hosp(
            random_variables=dict(predicted_hospitalizations=pred_hosps),
            constants=constants,
        )

        return IHR, pred_hosps, sampled_hosps
