# numpydoc ignore=GL08

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike

import pyrenew.arrayutils as au
from pyrenew.convolve import compute_delay_ascertained_incidence
from pyrenew.deterministic import DeterministicVariable
from pyrenew.metaclass import RandomVariable


class HospitalAdmissionsSample(NamedTuple):
    """
    A container to hold the output of `latent.HospAdmissions()`.

    Attributes
    ----------
    infection_hosp_rate : ArrayLike, optional
        The infection-to-hospitalization rate. Defaults to None.
    latent_hospital_admissions : ArrayLike or None
        The computed number of hospital admissions. Defaults to None.
    multiplier : ArrayLike or None
        The day of the week effect multiplier. Defaults to None. It
        should match the number of timepoints in the latent hospital
        admissions.
    """

    infection_hosp_rate: ArrayLike | None = None
    latent_hospital_admissions: ArrayLike | None = None
    multiplier: ArrayLike | None = None

    def __repr__(self):
        return f"HospitalAdmissionsSample(infection_hosp_rate={self.infection_hosp_rate}, latent_hospital_admissions={self.latent_hospital_admissions}, multiplier={self.multiplier})"


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
        infection_hospitalization_ratio_rv: RandomVariable,
        day_of_week_effect_rv: RandomVariable | None = None,
        hospitalization_reporting_ratio_rv: RandomVariable | None = None,
        obs_data_first_day_of_the_week: int = 0,
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        infection_to_admission_interval_rv : RandomVariable
            pmf for reporting (informing) hospital admissions (see
            pyrenew.observations.Deterministic).
        infection_hospitalization_ratio_rv : RandomVariable
            Infection to hospitalization rate random variable.
        day_of_week_effect_rv : RandomVariable, optional
            Day of the week effect. Should return a ArrayLike with 7
            values. Defaults to a deterministic variable with
            jax.numpy.ones(7) (no effect).
        hospitalization_reporting_ratio_rv  : RandomVariable, optional
            Random variable for the hospital admission reporting
            probability. Defaults to 1 (full reporting).
        obs_data_first_day_of_the_week : int, optional
            The day of the week that the first day of the observation data
            corresponds to. Valid values are 0-6, where 0 is Monday and 6 is
            Sunday. Defaults to 0.

        Returns
        -------
        None
        """

        if day_of_week_effect_rv is None:
            day_of_week_effect_rv = DeterministicVariable(
                name="weekday_effect", value=jnp.ones(7)
            )
        if hospitalization_reporting_ratio_rv is None:
            hospitalization_reporting_ratio_rv = DeterministicVariable(
                name="hosp_report_prob", value=1.0
            )

        HospitalAdmissions.validate(
            infection_to_admission_interval_rv,
            infection_hospitalization_ratio_rv,
            day_of_week_effect_rv,
            hospitalization_reporting_ratio_rv,
            obs_data_first_day_of_the_week,
        )

        self.infection_to_admission_interval_rv = (
            infection_to_admission_interval_rv
        )
        self.infection_hospitalization_ratio_rv = (
            infection_hospitalization_ratio_rv
        )
        self.day_of_week_effect_rv = day_of_week_effect_rv
        self.hospitalization_reporting_ratio_rv = (
            hospitalization_reporting_ratio_rv
        )
        self.obs_data_first_day_of_the_week = obs_data_first_day_of_the_week

    @staticmethod
    def validate(
        infection_to_admission_interval_rv: Any,
        infection_hospitalization_ratio_rv: Any,
        day_of_week_effect_rv: Any,
        hospitalization_reporting_ratio_rv: Any,
        obs_data_first_day_of_the_week: Any,
    ) -> None:
        """
        Validates that the IHR, weekday effects, probability of being
        reported hospitalized distributions, and infection to
        hospital admissions reporting delay pmf are RandomVariable types

        Parameters
        ----------
        infection_to_admission_interval_rv : Any
            Possibly incorrect input for the infection to hospitalization
            interval distribution.
        infection_hospitalization_ratio_rv : Any
            Possibly incorrect input for infection to hospitalization rate distribution.
        day_of_week_effect_rv : Any
            Possibly incorrect input for day of the week effect.
        hospitalization_reporting_ratio_rv : Any
            Possibly incorrect input for distribution or fixed value for the
            hospital admission reporting probability.
        obs_data_first_day_of_the_week : Any
            Possibly incorrect input for the day of the week that the first day
            of the observation data corresponds to. Valid values are 0-6, where
            0 is Monday and 6 is Sunday.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If any of the random variables are not of the correct type, or if
            the day of the week is not within the valid range.
        """
        assert isinstance(infection_to_admission_interval_rv, RandomVariable)
        assert isinstance(infection_hospitalization_ratio_rv, RandomVariable)
        assert isinstance(day_of_week_effect_rv, RandomVariable)
        assert isinstance(hospitalization_reporting_ratio_rv, RandomVariable)
        assert isinstance(obs_data_first_day_of_the_week, int)
        assert 0 <= obs_data_first_day_of_the_week <= 6

        return None

    def sample(
        self,
        latent_infections: ArrayLike,
        **kwargs,
    ) -> HospitalAdmissionsSample:
        """
        Samples from the observation process

        Parameters
        ----------
        latent_infections : ArrayLike
            Latent infections. Possibly the output of the `latent.Infections()`.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal `sample()`
            calls, should there be any.

        Returns
        -------
        HospitalAdmissionsSample
        """

        infection_hosp_rate, *_ = self.infection_hospitalization_ratio_rv(
            **kwargs
        )

        (
            infection_to_admission_interval,
            *_,
        ) = self.infection_to_admission_interval_rv(**kwargs)

        latent_hospital_admissions = compute_delay_ascertained_incidence(
            latent_infections,
            infection_to_admission_interval,
            infection_hosp_rate,
        )

        # Applying the day of the week effect. For this we need to:
        # 1. Get the day of the week effect
        # 2. Identify the offset of the latent_infections
        # 3. Apply the day of the week effect to the latent_hospital_admissions
        dow_effect_sampled = self.day_of_week_effect_rv(**kwargs, record=True)[
            0
        ]

        if dow_effect_sampled.size != 7:
            raise ValueError(
                "Day of the week effect should have 7 values. "
                f"Got {dow_effect_sampled.size} instead."
            )

        inf_offset = self.obs_data_first_day_of_the_week % 7

        # Replicating the day of the week effect to match the number of
        # timepoints
        dow_effect = au.tile_until_n(
            data=dow_effect_sampled,
            n_timepoints=latent_hospital_admissions.size,
            offset=inf_offset,
        )

        latent_hospital_admissions = latent_hospital_admissions * dow_effect

        # Applying reporting probability
        latent_hospital_admissions = (
            latent_hospital_admissions
            * self.hospitalization_reporting_ratio_rv(**kwargs)
        )

        numpyro.deterministic(
            "latent_hospital_admissions", latent_hospital_admissions
        )

        return HospitalAdmissionsSample(
            infection_hosp_rate=infection_hosp_rate,
            latent_hospital_admissions=latent_hospital_admissions,
            multiplier=dow_effect,
        )
