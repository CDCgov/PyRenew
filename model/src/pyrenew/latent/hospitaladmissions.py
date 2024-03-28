# -*- coding: utf-8 -*-

from collections import namedtuple

import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
from pyrenew.deterministic import DeterministicVariable
from pyrenew.metaclass import RandomVariable

HospAdmissionsSample = namedtuple(
    "HospAdmissionsSample",
    ["IHR", "predicted"],
    defaults=[None, None],
)
"""Output from HospitalAdmissions.sample()"""

InfectHospRateSample = namedtuple(
    "InfectHospRateSample",
    ["IHR"],
    defaults=[None],
)


class InfectHospRate(RandomVariable):
    """Infection to Hospitalization Rate"""

    def __init__(
        self,
        varname: str = "IHR",
        dist: dist.Distribution = dist.LogNormal(jnp.log(0.05), 0.05),
    ) -> None:
        """Default constructor

        Parameters
        ----------
        varname : str, optional
            Name of the random_variable that may hold observed IHR, by default
            "IHR"
        dist : dist.Distribution, optional
            Prior distribution of the IHR, by default
            dist.LogNormal(jnp.log(0.05), 0.05)

        Returns
        -------
        None
        """

        self.validate(dist)

        self.dist = dist
        self.varname = varname

        return None

    @staticmethod
    def validate(distr: dist.Distribution) -> None:
        assert isinstance(distr, dist.Distribution)

    def sample(
        self,
        random_variables: dict = None,
        constants: dict = None,
    ) -> InfectHospRateSample:
        return InfectHospRateSample(
            npro.sample(
                "IHR",
                self.dist,
                obs=random_variables.get(self.varname, None),
            )
        )


class HospitalAdmissions(RandomVariable):
    r"""Latent hospital admissions

    Implements a renewal process for the expected number of hospitalizations.

    Notes
    -----

    The following text was directly extracted from the wastewater model
    documentation
    (`link <https://github.com/cdcent/cfa-forecast-renewal-ww/blob/a17efc090b2ffbc7bc11bdd9eec5198d6bcf7322/model_definition.md#hospital-admissions-component> `_).

    Following other semi-mechanistic renewal frameworks, we model the _expected_
    hospital admissions per capita :math:`H(t)` as a convolution of the
    _expected_ latent incident infections per capita :math:`I(t)`, and a
    discrete infection to hospitalization distribution :math:`d(\tau)`, scaled
    by the probability of being hospitalized :math:`p_\mathrm{hosp}(t)`.

    To account for day-of-week effects in hospital reporting, we use an
    estimated _weekday effect_ :math:`\omega(t)`. If :math:`t` and :math:`t'`
    are the same day of the week, :math:`\omega(t) = \omega(t')`. The seven
    values that :math:`\omega(t)` takes on are constrained to have mean 1.

    .. math::

        H(t) = \omega(t) p_\mathrm{hosp}(t) \sum_{\\tau = 0}^{T_d} d(\tau) I(t-\tau)

    Where :math:`T_d` is the maximum delay from infection to hospitalization
    that we consider.
    """

    def __init__(
        self,
        inf_hosp_int: RandomVariable,
        infections_varname: str = "infections",
        hospitalizations_predicted_varname: str = "predicted_hospitalizations",
        infect_hosp_rate_dist: RandomVariable = InfectHospRate("IHR"),
        weekday_effect_dist: RandomVariable = DeterministicVariable((1,)),
        p_hosp_dist: RandomVariable = DeterministicVariable((1,)),
    ) -> None:
        """Default constructor

        Parameters
        ----------
        inf_hosp_int : RandomVariable
            pmf for reporting (informing) hospitalizations (see
            pyrenew.observations.Deterministic).
        infections_varname : str
            Name of the entry in random_variables that holds the vector of
            infections.
        infect_hosp_rate_varname : str
            Name of the entry in random_variables that holds the observed
            infection-hospitalization rate (IHR).
            (if available).
        hospitalizations_predicted_varname : str
            Name to assign to the deterministic component in numpyro of
            predicted hospitalizations.
        infect_hosp_rate_dist : RandomVariable, optional
            Infection to hospitalization rate pmf.
        weekday_effect_dist : RandomVariable, optional
            Weekday effect.
        p_hosp_dist : RandomVariable, optional
            Hospitalization probability.

        Returns
        -------
        None
        """
        HospitalAdmissions.validate(
            infect_hosp_rate_dist,
            weekday_effect_dist,
            p_hosp_dist,
        )

        self.infections_varname = infections_varname
        self.hospitalizations_predicted_varname = (
            hospitalizations_predicted_varname
        )

        self.infect_hosp_rate_dist = infect_hosp_rate_dist
        self.weekday_effect_dist = weekday_effect_dist
        self.p_hosp_dist = p_hosp_dist
        self.inf_hosp = inf_hosp_int

    @staticmethod
    def validate(
        infect_hosp_rate_dist, weekday_effect_dist, p_hosp_dist
    ) -> None:
        assert isinstance(infect_hosp_rate_dist, RandomVariable)
        assert isinstance(weekday_effect_dist, RandomVariable)
        assert isinstance(p_hosp_dist, RandomVariable)

        return None

    def sample(
        self,
        random_variables: dict = None,
        constants: dict = None,
    ) -> HospAdmissionsSample:
        """Samples from the observation process

        Parameters
        ----------
        random_variables : dict
            A dictionary `self.infections_varname` with the observed
            infections. Optionally, with IHR passed to obs in npyro.sample().
        constants : dict, optional
            Ignored.

        Returns
        -------
        HospAdmissionsSample
        """

        if random_variables is None:
            random_variables = dict()

        if constants is None:
            constants = dict()

        IHR, *_ = self.infect_hosp_rate_dist.sample(
            random_variables=random_variables,
            constants=constants,
        )

        IHR_t = IHR * random_variables.get(self.infections_varname)

        inf_hosp, *_ = self.inf_hosp.sample(
            random_variables=random_variables,
            constants=constants,
        )

        predicted_hospitalizations = jnp.convolve(
            IHR_t, inf_hosp, mode="full"
        )[: IHR_t.shape[0]]

        # Applying weekday effect
        predicted_hospitalizations = (
            predicted_hospitalizations
            * self.weekday_effect_dist.sample(
                random_variables=random_variables,
                constants=constants,
            )[0]
        )

        # Applying probability of hospitalization effect
        predicted_hospitalizations = (
            predicted_hospitalizations
            * self.p_hosp_dist.sample(
                random_variables=random_variables,
                constants=constants,
            )[0]
        )

        npro.deterministic(
            self.hospitalizations_predicted_varname, predicted_hospitalizations
        )

        return HospAdmissionsSample(IHR, predicted_hospitalizations)
