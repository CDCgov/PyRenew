# -*- coding: utf-8 -*-

from collections import namedtuple

from numpy.typing import ArrayLike
from pyrenew.deterministic import DeterministicVariable
from pyrenew.metaclass import Model, RandomVariable, _assert_sample_and_rtype
from pyrenew.model.rtinfectionsrenewalmodel import RtInfectionsRenewalModel

HospModelSample = namedtuple(
    "HospModelSample",
    [
        "Rt",
        "infections",
        "infection_hosp_rate",
        "latent",
        "sampled",
    ],
    defaults=[None, None, None, None, None],
)
HospModelSample.__doc__ = """
A container for holding the output from HospitalAdmissionsModel.sample().

Attributes
------HospitalAdmissionsModel
Rt : float or None
    The reproduction number over time. Defaults to None.
infections : ArrayLike or None
    The estimated number of new infections over time. Defaults to None.
infection_hosp_rate : float or None
    The infected hospitalization rate. Defaults to None.
latent : ArrayLike or None
    The estimated latent hospital admissions. Defaults to None.
sampled : ArrayLike or None
    The sampled or observed hospital admissions. Defaults to None.

Notes
-----
"""


class HospitalAdmissionsModel(Model):
    """
    Hospital Admissions Model (BasicRenewal + HospitalAdmissions)

    This class inherits from pyrenew.models.Model. It extends the
    basic renewal model by adding a hospital admissions module, e.g.,
    pyrenew.observations.HospitalAdmissions.
    """

    def __init__(
        self,
        latent_admissions: RandomVariable,
        latent_infections: RandomVariable,
        gen_int: RandomVariable,
        I0: RandomVariable,
        Rt_process: RandomVariable,
        observed_admissions: RandomVariable,
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        latent_admissions : RandomVariable
            Latent process for the hospital admissions.
        latent_infections : RandomVariable
            The infections latent process (passed to RtInfectionsRenewalModel).
        gen_int : RandomVariable
            Generation time (passed to RtInfectionsRenewalModel)
        HospitalAdmissionsModel
            Initial infections (passed to RtInfectionsRenewalModel)
        Rt_process : RandomVariable
            Rt process  (passed to RtInfectionsRenewalModel).
        observed_admissions : RandomVariable, optional
            Observation process for the hospital admissions.

        Returns
        -------
        None

        Notes
        -----
        TODO: See Also
        """
        self.basic_renewal = RtInfectionsRenewalModel(
            gen_int=gen_int,
            I0=I0,
            latent_infections=latent_infections,
            observed_infections=DeterministicVariable((0,)),
            Rt_process=Rt_process,
        )

        HospitalAdmissionsModel.validate(
            latent_admissions, observed_admissions
        )

        self.latent_admissions = latent_admissions
        self.observed_admissions = observed_admissions

    @staticmethod
    def validate(latent_admissions, observed_admissions) -> None:
        """
        Verifies types and status (RV) of latent and observed hospital admissions

        Parameters
        ----------
        latent_admissions : ArrayLike
            The latent process for the hospial admissions.
        observed_admissions : ArrayLike
            The observed hospital admissions.

        Returns
        -------
        None

        See Also
        --------
        _assert_sample_and_rtype : Perform type-checking and verify RV
        """
        _assert_sample_and_rtype(latent_admissions, skip_if_none=False)
        _assert_sample_and_rtype(observed_admissions, skip_if_none=False)
        return None

    def sample_latent_admissions(
        self,
        infections: ArrayLike,
        **kwargs,
    ) -> tuple:
        """
        Sample number of hospital admissions

        Parameters
        ----------
        infections : ArrayLike
            The predicted infections array.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample() calls, should there be any.

        Returns
        -------
        tuple

        See Also
        --------
        latent_admissions.sample : For sampling latent hospital admissions

        Notes
        -----
        TODO: Include example(s) here.
        TODO: Cover Returns in more detail.
        """

        return self.latent_admissions.sample(
            latent=infections,
            **kwargs,
        )

    def sample_observed_admissions(
        self,
        predicted: ArrayLike,
        observed_admissions: ArrayLike,
        **kwargs,
    ) -> tuple:
        """
        Sample number of hospital admissions

        Parameters
        ----------
        predicted : ArrayLike
            The predicted hospital admissions.
        observed_admissions : ArrayLike
            The observed hospitalization data (to fit).
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample_observed_admissions calls, should there be any.

        Returns
        -------
        tuple

        See Also
        --------
        observed_admissions.sample : For sampling observed hospital
        admissions.

        Notes
        -----
        TODO: Include example(s) here.
        """

        return self.observed_admissions.sample(
            predicted=predicted, obs=observed_admissions, **kwargs
        )

    def sample(
        self,
        n_timepoints: int,
        observed_admissions: ArrayLike | None = None,
        **kwargs,
    ) -> HospModelSample:
        """
        Sample from the HospitalAdmissions model

        Parameters
        ----------
        n_timepoints : int
            Number of timepoints to sample (passed to the basic renewal model).
        observed_admissions : ArrayLike, optional
            The observed hospitalization data (passed to the basic renewal
            model). Defaults to None (simulation, rather than fit).
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, should there be any.

        Returns
        -------
        HospModelSample

        See Also
        --------
        basic_renewal.sample : For sampling the basic renewal model
        sample_latent_admissions : To sample latent hospitalization process
        sample_observed_admissions : For sampling observed hospital admissions

        Notes
        -----
        TODO: Include example(s) here.
        """

        # Getting the initial quantities from the basic model
        Rt, infections, *_ = self.basic_renewal.sample(
            n_timepoints=n_timepoints,
            observed_infections=None,
            **kwargs,
        )

        # Sampling the latent hospital admissions
        (
            infection_hosp_rate,
            latent,
            *_,
        ) = self.sample_latent_admissions(
            infections=infections,
            **kwargs,
        )

        # Sampling the hospital admissions
        sampled, *_ = self.sample_observed_admissions(
            predicted=latent,
            observed_admissions=observed_admissions,
            **kwargs,
        )

        return HospModelSample(
            Rt=Rt,
            infections=infections,
            infection_hosp_rate=infection_hosp_rate,
            latent=latent,
            sampled=sampled,
        )
