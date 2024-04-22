# -*- coding: utf-8 -*-

from collections import namedtuple
from typing import Any, Optional

import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
from numpy.typing import ArrayLike
from pyrenew.deterministic import DeterministicVariable
from pyrenew.metaclass import RandomVariable

HospAdmissionsSample = namedtuple(
    "HospAdmissionsSample",
    ["infection_hosp_rate", "predicted"],
    defaults=[None, None],
)
HospAdmissionsSample.__doc__ = """
A container for holding the output from HospAdmissionsSample.sample.

Attributes
----------
infection_hosp_rate : float or None
    The infected hospitalization rate. Defaults to None.
predicted : ArrayLike or None
    The predicted number of hospital admissions. Defaults to None.

Notes
-----
TODO: Add Notes.
"""

InfectHospRateSample = namedtuple(
    "InfectHospRateSample",
    ["infection_hosp_rate"],
    defaults=[None],
)
InfectHospRateSample.__doc__ = """
A container for holding the output from InfectHospRateSample.sample.

Attributes
----------
infection_hosp_rate : ArrayLike or None
    The infected hospitalization rate. Defaults to None.

Notes
-----
TODO: Add Notes.
"""


class InfectHospRate(RandomVariable):
    """
    Infection to Hospitalization Rate

    Methods
    -------
    validate(distr)
        Validates distribution is Numpyro distribution
    sample(**kwargs)
        Produces a sample of the infection-hospitalization rate (IHR).
    """

    def __init__(
        self,
        dist: dist.Distribution | None,
        varname: Optional[str] = "infection_hosp_rate",
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        dist : dist.Distribution
            Prior distribution of the IHR.
        varname : str, optional
            Name of the random variable in the model, by default "infection_hosp_rate."

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
        """
        Validates distribution is Numpyro distribution

        Parameters
        ----------
        distr : dist.Distribution
            A ingested distribution (e.g., prior IHR distribution)

        Raises
        ------
        AssertionError
            If the inputted distribution is not a Numpyro distribution.
        """
        assert isinstance(distr, dist.Distribution)

    def sample(self, **kwargs) -> InfectHospRateSample:
        """
        Produces a sample of the Infection-hospitalization Rate (IHR)

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample calls, should there be any.

        Returns
        -------
        InfectHospRateSample
            The sampled IHR
        """
        return InfectHospRateSample(
            npro.sample(
                name=self.varname,
                fn=self.dist,
            )
        )


class HospitalAdmissions(RandomVariable):
    r"""Latent hospital admissions

    Implements a renewal process for the expected number of hospital admissions.

    Methods
    -------
    validate(infect_hosp_rate_dist, weekday_effect_dist, hosp_report_prob_dist)
        Validates that the IHR, weekday effects, and probability of being
        reported hospitalized distributions are RandomVariable types
    sample(latent, **kwargs)
        Samples from the observation process

    Notes
    -----
    The following text was directly extracted from the wastewater model
    documentation (`link <https://github.com/cdcent/cfa-forecast-renewal-ww/blob/a17efc090b2ffbc7bc11bdd9eec5198d6bcf7322/model_definition.md#hospital-admissions-component> `_).

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
        admissions_predicted_varname: str = "predicted_admissions",
        weekday_effect_dist: Optional[RandomVariable] = None,
        hosp_report_prob_dist: Optional[RandomVariable] = None,
    ) -> None:
        """Default constructor

        Parameters
        ----------
        infection_to_admission_interval : RandomVariable
            pmf for reporting (informing) hospital admissions (see
            pyrenew.observations.Deterministic).
        infect_hosp_rate_dist : RandomVariable
            Infection to hospitalization rate distribution.
        admissions_predicted_varname : str
            Name to assign to the deterministic component in numpyro of
            predicted hospital admissions.
        weekday_effect_dist : RandomVariable, optional
            Weekday effect.
        hosp_report_prob_dist  : RandomVariable, optional
            Distribution or fixed value for the hospital admission reporting
            probability. Defaults to 1 (full reporting).

        Returns
        -------
        None
        """

        if weekday_effect_dist is None:
            weekday_effect_dist = DeterministicVariable((1,))
        if hosp_report_prob_dist is None:
            hosp_report_prob_dist = DeterministicVariable((1,))

        HospitalAdmissions.validate(
            infect_hosp_rate_dist,
            weekday_effect_dist,
            hosp_report_prob_dist,
        )

        self.admissions_predicted_varname = admissions_predicted_varname

        self.infect_hosp_rate_dist = infect_hosp_rate_dist
        self.weekday_effect_dist = weekday_effect_dist
        self.hosp_report_prob_dist = hosp_report_prob_dist
        self.infection_to_admission_interval = infection_to_admission_interval

    @staticmethod
    def validate(
        infect_hosp_rate_dist: Any,
        weekday_effect_dist: Any,
        hosp_report_prob_dist: Any,
    ) -> None:
        """
        Validates that the IHR, weekday effects, and probability of being
        reported hospitalized distributions are RandomVariable types

        Parameters
        ----------
        infect_hosp_rate_dist : Any
            Possibly incorrect input for infection to hospitalization rate distribution.
        weekday_effect_dist : Any
            Possibly incorrect input for weekday effect.
        hosp_report_prob_dist  : Any
            Possibly incorrect input for distribution or fixed value for the
            hospital admission reporting probability.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the object `distr` is not an instance of `dist.Distribution`, indicating
            that the validation has failed.
        """
        assert isinstance(infect_hosp_rate_dist, RandomVariable)
        assert isinstance(weekday_effect_dist, RandomVariable)
        assert isinstance(hosp_report_prob_dist, RandomVariable)

        return None

    def sample(
        self,
        latent: ArrayLike,
        **kwargs,
    ) -> HospAdmissionsSample:
        """Samples from the observation process

        Parameters
        ----------
        latent : ArrayLike
            Latent infections.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal `sample()`
            calls, if any

        Returns
        -------
        HospAdmissionsSample
        """

        infection_hosp_rate, *_ = self.infect_hosp_rate_dist.sample(**kwargs)

        infection_hosp_rate_t = infection_hosp_rate * latent

        (
            infection_to_admission_interval,
            *_,
        ) = self.infection_to_admission_interval.sample(**kwargs)

        predicted_admissions = jnp.convolve(
            infection_hosp_rate_t, infection_to_admission_interval, mode="full"
        )[: infection_hosp_rate_t.shape[0]]

        # Applying weekday effect
        predicted_admissions = (
            predicted_admissions * self.weekday_effect_dist.sample(**kwargs)[0]
        )

        # Applying probability of hospitalization effect
        predicted_admissions = (
            predicted_admissions
            * self.hosp_report_prob_dist.sample(**kwargs)[0]
        )

        npro.deterministic(
            self.admissions_predicted_varname, predicted_admissions
        )

        return HospAdmissionsSample(infection_hosp_rate, predicted_admissions)
