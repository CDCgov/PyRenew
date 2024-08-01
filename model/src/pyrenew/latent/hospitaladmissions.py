# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpyro
from pyrenew.deterministic import DeterministicVariable
from pyrenew.metaclass import RandomVariable, SampledValue


class HospitalAdmissionsSample(NamedTuple):
    """
    A container to hold the output of `latent.HospAdmissions()`.

    Attributes
    ----------
    infection_hosp_rate : SampledValue, optional
        The infection-to-hospitalization rate. Defaults to None.
    latent_hospital_admissions : SampledValue or None
        The computed number of hospital admissions. Defaults to None.
    """

    infection_hosp_rate: SampledValue | None = None
    latent_hospital_admissions: SampledValue | None = None

    def __repr__(self):
        return f"HospitalAdmissionsSample(infection_hosp_rate={self.infection_hosp_rate}, latent_hospital_admissions={self.latent_hospital_admissions})"


class HospitalAdmissions(RandomVariable):
    r"""
    Latent hospital admissions

    Implements a renewal process for the expected number of hospital admissions.

    Notes
    -----
    The following text was directly extracted from the wastewater model
    documentation (`link <https://github.com/cdcent/cfa-forecast-renewal-ww/blob/a17efc090b2ffbc7bc11bdd9eec5198d6bcf7322/model_definition.md#hospital-admissions-component>`_).

    Following other semi-mechanistic renewal frameworks, we model the *expected*
    hospital admissions per capita :math:`H(t)` as a convolution of the
    *expected* latent incident infections per capita :math:`I(t)`, and a
    discrete infection to hospitalization distribution :math:`d(\tau)`, scaled
    by the probability of being hospitalized :math:`p_\mathrm{hosp}(t)`.

    To account for day-of-week effects in hospital reporting, we use an
    estimated *day of the week effect* :math:`\omega(t)`. If :math:`t` and :math:`t'`
    are the same day of the week, :math:`\omega(t) = \omega(t')`. The seven
    values that :math:`\omega(t)` takes on are constrained to have mean 1.

    .. math::

        H(t) = \omega(t) p_\mathrm{hosp}(t) \sum_{\tau = 0}^{T_d} d(\tau) I(t-\tau)

    Where :math:`T_d` is the maximum delay from infection to hospitalization
    that we consider.
    """

    def __init__(
        self,
        infection_to_admission_interval_rv: RandomVariable,
        infect_hosp_rate_rv: RandomVariable,
        day_of_week_effect_rv: RandomVariable | None = None,
        hosp_report_prob_rv: RandomVariable | None = None,
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        infection_to_admission_interval_rv : RandomVariable
            pmf for reporting (informing) hospital admissions (see
            pyrenew.observations.Deterministic).
        infect_hosp_rate_rv : RandomVariable
            Infection to hospitalization rate random variable.
        day_of_week_effect_rv : RandomVariable, optional
            Day of the week effect.
        hosp_report_prob_rv  : RandomVariable, optional
            Random variable for the hospital admission reporting
            probability. Defaults to 1 (full reporting).

        Returns
        -------
        None
        """

        if day_of_week_effect_rv is None:
            day_of_week_effect_rv = DeterministicVariable(
                name="weekday_effect", value=1
            )
        if hosp_report_prob_rv is None:
            hosp_report_prob_rv = DeterministicVariable(
                name="hosp_report_prob", value=1
            )

        HospitalAdmissions.validate(
            infect_hosp_rate_rv,
            day_of_week_effect_rv,
            hosp_report_prob_rv,
        )

        self.infect_hosp_rate_rv = infect_hosp_rate_rv
        self.day_of_week_effect_rv = day_of_week_effect_rv
        self.hosp_report_prob_rv = hosp_report_prob_rv
        self.infection_to_admission_interval_rv = (
            infection_to_admission_interval_rv
        )
        # Why isn't infection_to_admission_interval_rv validated?

    @staticmethod
    def validate(
        infect_hosp_rate_rv: Any,
        day_of_week_effect_rv: Any,
        hosp_report_prob_rv: Any,
    ) -> None:
        """
        Validates that the IHR, weekday effects, and probability of being
        reported hospitalized distributions are RandomVariable types

        Parameters
        ----------
        infect_hosp_rate_rv : Any
            Possibly incorrect input for infection to hospitalization rate distribution.
        day_of_week_effect_rv : Any
            Possibly incorrect input for day of the week effect.
        hosp_report_prob_rv : Any
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
        assert isinstance(infect_hosp_rate_rv, RandomVariable)
        assert isinstance(day_of_week_effect_rv, RandomVariable)
        assert isinstance(hosp_report_prob_rv, RandomVariable)

        return None

    def sample(
        self,
        latent_infections: SampledValue,
        **kwargs,
    ) -> HospitalAdmissionsSample:
        """
        Samples from the observation process

        Parameters
        ----------
        latent_infections : ArrayLike
            Latent infections.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal `sample()`
            calls, should there be any.

        Returns
        -------
        HospitalAdmissionsSample
        """

        infection_hosp_rate, *_ = self.infect_hosp_rate_rv(**kwargs)

        infection_hosp_rate_t = (
            infection_hosp_rate.value * latent_infections.value
        )

        (
            infection_to_admission_interval,
            *_,
        ) = self.infection_to_admission_interval_rv(**kwargs)

        latent_hospital_admissions = jnp.convolve(
            infection_hosp_rate_t,
            infection_to_admission_interval.value,
            mode="full",
        )[: infection_hosp_rate_t.shape[0]]

        # Applying the day of the week effect
        latent_hospital_admissions = (
            latent_hospital_admissions
            * self.day_of_week_effect_rv(
                obs=SampledValue(
                    latent_hospital_admissions,
                    t_start=latent_infections.t_start,
                ),
                **kwargs,
            )[0].value
        )

        # Applying reporting probability
        latent_hospital_admissions = (
            latent_hospital_admissions
            * self.hosp_report_prob_rv(**kwargs)[0].value
        )

        numpyro.deterministic(
            "latent_hospital_admissions", latent_hospital_admissions
        )

        return HospitalAdmissionsSample(
            infection_hosp_rate=infection_hosp_rate,
            latent_hospital_admissions=SampledValue(
                value=latent_hospital_admissions,
                t_start=self.t_start,
                t_unit=self.t_unit,
            ),
        )
