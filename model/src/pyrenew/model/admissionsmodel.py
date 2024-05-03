# -*- coding: utf-8 -*-


from typing import NamedTuple

import jax.numpy as jnp
from numpy.typing import ArrayLike
from pyrenew.metaclass import Model, RandomVariable, _assert_sample_and_rtype
from pyrenew.model.rtinfectionsrenewalmodel import RtInfectionsRenewalModel


class HospModelSample(NamedTuple):
    """
    A container for holding the output from HospitalAdmissionsModel.sample().

    Attributes
    ----------
    Rt : float | None, optional
        The reproduction number over time. Defaults to None.
    latent_infections : ArrayLike | None, optional
        The estimated number of new infections over time. Defaults to None.
    IHR : float | None, optional
        The infected hospitalization rate. Defaults to None.
    latent_admissions : ArrayLike | None, optional
        The estimated latent hospitalizations. Defaults to None.
    sampled_admissions : ArrayLike | None, optional
        The sampled or observed hospital admissions. Defaults to None.
    """

    Rt: float | None = None
    latent_infections: ArrayLike | None = None
    IHR: float | None = None
    latent_admissions: ArrayLike | None = None
    sampled_admissions: ArrayLike | None = None

    def __repr__(self):
        return f"HospModelSample(Rt={self.Rt}, latent_infections={self.latent_infections}, IHR={self.IHR}, latent_admissions={self.latent_admissions}, sampled_admissions={self.sampled_admissions})"


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
        observation_process: RandomVariable,
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
        observation_process : RandomVariable, optional
            Observation process for the hospital admissions.

        Returns
        -------
        None
        """
        self.basic_renewal = RtInfectionsRenewalModel(
            gen_int=gen_int,
            I0=I0,
            latent_infections=latent_infections,
            observation_process=None,
            Rt_process=Rt_process,
        )

        HospitalAdmissionsModel.validate(
            latent_admissions, observation_process
        )

        self.latent_admissions = latent_admissions
        self.observation_process = observation_process

    @staticmethod
    def validate(latent_admissions, observation_process) -> None:
        """
        Verifies types and status (RV) of latent and observed hospital admissions

        Parameters
        ----------
        latent_admissions : ArrayLike
            The latent process for the hospital admissions.
        observation_process : ArrayLike
            The observed hospital admissions.

        Returns
        -------
        None

        See Also
        --------
        _assert_sample_and_rtype : Perform type-checking and verify RV
        """
        _assert_sample_and_rtype(latent_admissions, skip_if_none=False)
        _assert_sample_and_rtype(observation_process, skip_if_none=False)
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

    def sample_admissions_process(
        self,
        predicted: ArrayLike,
        observed_admissions: ArrayLike,
        name: str | None = None,
        **kwargs,
    ) -> tuple:
        """
        Sample number of hospital admissions

        Parameters
        ----------
        predicted : ArrayLike
            The predicted hospital admissions.
        obs : ArrayLike
            The observed hospitalization data (to fit).
        name : str, optional
            Name of the random variable. Defaults to None.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            obs calls, should there be any.

        Returns
        -------
        tuple
        """

        return self.observation_process.sample(
            predicted=predicted,
            obs=observed_admissions,
            name=name,
            **kwargs,
        )

    def sample(
        self,
        n_timepoints: int,
        observed_admissions: ArrayLike | None = None,
        padding: int = 0,
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
        padding : int, optional
            Number of padding timepoints to add to the beginning of the
            simulation. Defaults to 0.
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
        basic_model = self.basic_renewal.sample(
            n_timepoints=n_timepoints,
            observed_infections=None,
            padding=padding,
            **kwargs,
        )

        # Sampling the latent hospital admissions
        (
            infection_hosp_rate,
            latent,
            *_,
        ) = self.sample_latent_admissions(
            infections=basic_model.latent_infections,
            **kwargs,
        )

        # Sampling the hospital admissions
        if self.observation_process is not None:
            if (observed_admissions is not None) and (padding > 0):
                sampled_na = jnp.repeat(jnp.nan, padding)

                sampled_observed, *_ = self.sample_admissions_process(
                    predicted=latent[padding:],
                    observed_admissions=observed_admissions[padding:],
                    **kwargs,
                )

                sampled = jnp.hstack([sampled_na, sampled_observed])

            else:
                sampled, *_ = self.sample_admissions_process(
                    predicted=latent,
                    observed_admissions=observed_admissions,
                    **kwargs,
                )
        else:
            sampled = None

        return HospModelSample(
            Rt=basic_model.Rt,
            latent_infections=basic_model.latent_infections,
            IHR=infection_hosp_rate,
            latent_admissions=latent,
            sampled_admissions=sampled,
        )
