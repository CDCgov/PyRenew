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
        dist: dist.Distribution,
        varname: str = "IHR",
    ) -> None:
        """Default constructor

        Parameters
        ----------
        dist : dist.Distribution, optional
            Prior distribution of the IHR, by default
            dist.LogNormal(jnp.log(0.05), 0.05)
        varname : str, optional
            Name of the random_variable that may hold observed IHR, by default
            "IHR"

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
        infection_to_admission_interval: RandomVariable,
        infect_hosp_rate_dist: RandomVariable,
        infections_varname: str = "infections",
        hospitalizations_predicted_varname: str = "predicted_hospitalizations",
        weekday_effect_dist: RandomVariable = DeterministicVariable((1,)),
        hospitalizations_reporting_dist: RandomVariable = DeterministicVariable(
            (1,)
        ),
    ) -> None:
        """Default constructor

        Parameters
        ----------
        infection_to_admission_interval : RandomVariable
            pmf for reporting (informing) hospitalizations (see
            pyrenew.observations.Deterministic).
        infect_hosp_rate_dist : RandomVariable
            Infection to hospitalization rate distribution.
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
        weekday_effect_dist : RandomVariable, optional
            Weekday effect.
        hospitalizations_reporting_dist  : RandomVariable, optional
            Reporting probability for hospital admissions. Defaults to 1 (full
            reporting).

        Returns
        -------
        None
        """
        HospitalAdmissions.validate(
            infect_hosp_rate_dist,
            weekday_effect_dist,
            hospitalizations_reporting_dist,
        )

        self.infections_varname = infections_varname
        self.hospitalizations_predicted_varname = (
            hospitalizations_predicted_varname
        )

        self.infect_hosp_rate_dist = infect_hosp_rate_dist
        self.weekday_effect_dist = weekday_effect_dist
        self.hospitalizations_reporting_dist = hospitalizations_reporting_dist
        self.infection_to_admission_interval = infection_to_admission_interval

    @staticmethod
    def validate(
        infect_hosp_rate_dist,
        weekday_effect_dist,
        hospitalizations_reporting_dist,
    ) -> None:
        assert isinstance(infect_hosp_rate_dist, RandomVariable)
        assert isinstance(weekday_effect_dist, RandomVariable)
        assert isinstance(hospitalizations_reporting_dist, RandomVariable)

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

        (
            infection_to_admission_interval,
            *_,
        ) = self.infection_to_admission_interval.sample(
            random_variables=random_variables,
            constants=constants,
        )

        predicted_hospitalizations = jnp.convolve(
            IHR_t, infection_to_admission_interval, mode="full"
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
            * self.hospitalizations_reporting_dist.sample(
                random_variables=random_variables,
                constants=constants,
            )[0]
        )

        npro.deterministic(
            self.hospitalizations_predicted_varname, predicted_hospitalizations
        )

        return HospAdmissionsSample(IHR, predicted_hospitalizations)
