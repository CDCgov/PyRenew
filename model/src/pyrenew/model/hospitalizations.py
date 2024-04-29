# -*- coding: utf-8 -*-


import jax.numpy as jnp
from numpy.typing import ArrayLike
from pyrenew.metaclass import Model, RandomVariable, _assert_sample_and_rtype
from pyrenew.model.rtinfectionsrenewal import RtInfectionsRenewalModel


class HospModelSample:
    """
    A container for holding the output from HospitalAdmissionsModel.sample().
    """

    def __init__(
        self,
        Rt=None,
        latent_infections=None,
        IHR=None,
        latent_admissions=None,
        sampled_admissions=None,
    ) -> None:
        """
        Default constructor

        Parameters
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

        Returns
        -------
        None
        """
        self.Rt = Rt
        self.latent_infections = latent_infections
        self.IHR = IHR
        self.latent_admissions = latent_admissions
        self.sampled_admissions = sampled_admissions

    def __repr__(self):
        return f"HospModelSample(Rt={self.Rt}, latent_infections={self.latent_infections}, IHR={self.IHR}, latent_admissions={self.latent_admissions}, sampled_admissions={self.sampled_admissions})"


class HospitalizationsModel(Model):
    """
    HospitalAdmissions Model (BasicRenewal + HospitalAdmissions)

    This class inherits from pyrenew.models.Model. It extends the
    basic renewal model by adding a hospital admissions module, e.g.,
    pyrenew.observations.HospitalAdmissions.
    """

    def __init__(
        self,
        latent_hospitalizations: RandomVariable,
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
        latent_hospitalizations : RandomVariable
            Latent process for the hospitalizations.
        latent_infections : RandomVariable
            The infections latent process (passed to RtInfectionsRenewalModel).
        gen_int : RandomVariable
            Generation time (passed to RtInfectionsRenewalModel)
        I0 : RandomVariable
            Initial infections (passed to RtInfectionsRenewalModel)
        Rt_process : RandomVariable
            Rt process  (passed to RtInfectionsRenewalModel).
        observation_process : RandomVariable, optional
            Observation process for the hospitalizations.

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

        HospitalizationsModel.validate(
            latent_hospitalizations, observation_process
        )

        self.latent_hospitalizations = latent_hospitalizations
        self.observation_process = observation_process

    @staticmethod
    def validate(latent_hospitalizations, observation_process) -> None:
        """
        Verifies types and status (RV) of latent and observed hospitalizations

        Parameters
        ----------
        latent_hospitalizations : ArrayLike
            The latent process for the hospitalizations.
        observation_process : ArrayLike
            The observed hospitalizations.

        Returns
        -------
        None

        See Also
        --------
        _assert_sample_and_rtype : Perform type-checking and verify RV
        """
        _assert_sample_and_rtype(latent_hospitalizations, skip_if_none=False)
        _assert_sample_and_rtype(observation_process, skip_if_none=False)
        return None

    def sample_hospitalizations_latent(
        self,
        infections: ArrayLike,
        **kwargs,
    ) -> tuple:
        """Sample number of hospitalizations

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
        latent_hospitalizations.sample : For sampling latent hospitalizations
        """

        return self.latent_hospitalizations.sample(
            latent=infections,
            **kwargs,
        )

    def sample_hospitalizations_obs(
        self,
        predicted: ArrayLike,
        observed_hospitalizations: ArrayLike,
        name: str | None = None,
        **kwargs,
    ) -> tuple:
        """Sample number of hospitalizations

        Parameters
        ----------
        predicted : ArrayLike
            The predicted hospitalizations.
        observed_hospitalizations : ArrayLike
            The observed hospitalization data (to fit).
        name : str, optional
            Name of the random variable. Defaults to None.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample_hospitalizations_obs calls, should there be any.

        Returns
        -------
        tuple
        """

        return self.observation_process.sample(
            predicted=predicted,
            obs=observed_hospitalizations,
            name=name,
            **kwargs,
        )

    def sample(
        self,
        n_timepoints: int,
        observed_hospitalizations: ArrayLike | None = None,
        padding: int = 0,
        **kwargs,
    ) -> HospModelSample:
        """
        Sample from the HospitalAdmissions model

        Parameters
        ----------
        n_timepoints : int
            Number of timepoints to sample (passed to the basic renewal model).
        observed_hospitalizations : ArrayLike, optional
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
        sample_hospitalizations_latent : To sample latent hospitalization process
        sample_hospitalizations_obs : For sampling observed hospitalizations
        """

        # Getting the initial quantities from the basic model
        basic_model = self.basic_renewal.sample(
            n_timepoints=n_timepoints,
            observed_infections=None,
            padding=padding,
            **kwargs,
        )

        # Sampling the latent hospitalizations
        (
            IHR,
            latent,
            *_,
        ) = self.sample_hospitalizations_latent(
            infections=basic_model.latent_infections,
            **kwargs,
        )

        # Sampling the hospitalizations
        if self.observation_process is not None:
            if (observed_hospitalizations is not None) and (padding > 0):
                sampled_na = jnp.repeat(jnp.nan, padding)

                sampled_observed, *_ = self.sample_hospitalizations_obs(
                    predicted=latent[padding:],
                    observed_hospitalizations=observed_hospitalizations[
                        padding:
                    ],
                    **kwargs,
                )

                sampled = jnp.hstack([sampled_na, sampled_observed])

            else:
                sampled, *_ = self.sample_hospitalizations_obs(
                    predicted=latent,
                    observed_hospitalizations=observed_hospitalizations,
                    **kwargs,
                )
        else:
            sampled = None

        return HospModelSample(
            Rt=basic_model.Rt,
            latent_infections=basic_model.latent_infections,
            IHR=IHR,
            latent_admissions=latent,
            sampled_admissions=sampled,
        )
