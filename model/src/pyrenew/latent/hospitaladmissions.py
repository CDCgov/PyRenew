# -*- coding: utf-8 -*-


from typing import Any, Optional

import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
from numpy.typing import ArrayLike
from pyrenew.deterministic import DeterministicVariable
from pyrenew.metaclass import RandomVariable


class HospAdmissionsSample:
    """
    A container for holding the output from HospAdmissionsSample.sample.
    """

    def __init__(self, IHR=None, predicted=None) -> None:
        """
        Default constructor.

        Parameters
        ----------
        IHR : float, optional
            The infected hospitalization rate. Defaults to None.
        predicted : ArrayLike or None
            The predicted number of hospital admissions. Defaults to None.

        Returns
        -------
        None
        """
        self.IHR = IHR
        self.predicted = predicted

    def __repr__(self):
        return (
            f"HospAdmissionsSample(IHR={self.IRH}, predicted={self.predicted})"
        )


class InfectHospRateSample:
    """
    A container for holding the output from InfectHospRateSample.sample.
    """

    def __init__(self, IHR=None) -> None:
        """
        Default constructor

        Parameters
        ----------
        IHR : float, optional
            The infected hospitalization rate. Defaults to None.

        """
        self.IHR = IHR

    def __repr__(self):
        return f"InfectHospRateSample(IHR={self.IHR})"


class InfectHospRate(RandomVariable):
    """
    Infection to Hospitalization Rate
    """

    def __init__(
        self,
        dist: dist.Distribution | None,
        varname: Optional[str] = "IHR",
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        dist : dist.Distribution
            Prior distribution of the IHR.
        varname : str, optional
            Name of the random variable in the model, by default "IHR."

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
        Produces a sample of the IHR

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

    Implements a renewal process for the expected number of hospitalizations.


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

        H(t) = \omega(t) p_\mathrm{hosp}(t) \sum_{\tau = 0}^{T_d} d(\tau) I(t-\tau)

    Where :math:`T_d` is the maximum delay from infection to hospitalization
    that we consider.
    """

    def __init__(
        self,
        infection_to_admission_interval: RandomVariable,
        infect_hosp_rate_dist: RandomVariable,
        hospitalizations_predicted_varname: str = "predicted_hospitalizations",
        weekday_effect_dist: Optional[RandomVariable] = None,
        hosp_report_prob_dist: Optional[RandomVariable] = None,
    ) -> None:
        """Default constructor

        Parameters
        ----------
        infection_to_admission_interval : RandomVariable
            pmf for reporting (informing) hospitalizations (see
            pyrenew.observations.Deterministic).
        infect_hosp_rate_dist : RandomVariable
            Infection to hospitalization rate distribution.
        hospitalizations_predicted_varname : str
            Name to assign to the deterministic component in numpyro of
            predicted hospitalizations.
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
            weekday_effect_dist = DeterministicVariable(1)
        if hosp_report_prob_dist is None:
            hosp_report_prob_dist = DeterministicVariable(1)

        HospitalAdmissions.validate(
            infect_hosp_rate_dist,
            weekday_effect_dist,
            hosp_report_prob_dist,
        )

        self.hospitalizations_predicted_varname = (
            hospitalizations_predicted_varname
        )

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
            calls, should there be any.

        Returns
        -------
        HospAdmissionsSample
        """

        IHR, *_ = self.infect_hosp_rate_dist.sample(**kwargs)

        IHR_t = IHR * latent

        (
            infection_to_admission_interval,
            *_,
        ) = self.infection_to_admission_interval.sample(**kwargs)

        predicted_hospitalizations = jnp.convolve(
            IHR_t, infection_to_admission_interval, mode="full"
        )[: IHR_t.shape[0]]

        # Applying weekday effect
        predicted_hospitalizations = (
            predicted_hospitalizations
            * self.weekday_effect_dist.sample(**kwargs)[0]
        )

        # Applying probability of hospitalization effect
        predicted_hospitalizations = (
            predicted_hospitalizations
            * self.hosp_report_prob_dist.sample(**kwargs)[0]
        )

        npro.deterministic(
            self.hospitalizations_predicted_varname, predicted_hospitalizations
        )

        return HospAdmissionsSample(IHR, predicted_hospitalizations)
