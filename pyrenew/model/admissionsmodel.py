# numpydoc ignore=GL08

from __future__ import annotations

from typing import NamedTuple

from jax.typing import ArrayLike

from pyrenew.deterministic import NullObservation
from pyrenew.metaclass import Model, RandomVariable
from pyrenew.model.rtinfectionsrenewalmodel import RtInfectionsRenewalModel


class HospModelSample(NamedTuple):
    """
    A container for holding the output from `model.HospitalAdmissionsModel.sample()`.

    Attributes
    ----------
    Rt : ArrayLike | None, optional
        The reproduction number over time. Defaults to None.
    latent_infections : ArrayLike | None, optional
        The estimated number of new infections over time. Defaults to None.
    infection_hosp_rate : ArrayLike | None, optional
        The infected hospitalization rate. Defaults to None.
    latent_hosp_admissions : ArrayLike | None, optional
        The estimated latent hospitalizations. Defaults to None.
    observed_hosp_admissions : ArrayLike | None, optional
        The sampled or observed hospital admissions. Defaults to None.
    """

    Rt: ArrayLike | None = None
    latent_infections: ArrayLike | None = None
    infection_hosp_rate: ArrayLike | None = None
    latent_hosp_admissions: ArrayLike | None = None
    observed_hosp_admissions: ArrayLike | None = None

    def __repr__(self):
        return (
            f"HospModelSample(Rt={self.Rt}, "
            f"latent_infections={self.latent_infections}, "
            f"infection_hosp_rate={self.infection_hosp_rate}, "
            f"latent_hosp_admissions={self.latent_hosp_admissions}, "
            f"observed_hosp_admissions={self.observed_hosp_admissions}"
        )


class HospitalAdmissionsModel(Model):
    """
    Hospital Admissions Model (BasicRenewal + HospitalAdmissions)

    This class inherits from pyrenew.models.Model. It extends the
    basic renewal model by adding a hospital admissions module, e.g.,
    pyrenew.observations.HospitalAdmissions.
    """

    def __init__(
        self,
        latent_hosp_admissions_rv: RandomVariable,
        latent_infections_rv: RandomVariable,
        gen_int_rv: RandomVariable,
        I0_rv: RandomVariable,
        Rt_process_rv: RandomVariable,
        hosp_admission_obs_process_rv: RandomVariable,
    ) -> None:  # numpydoc ignore=PR04
        """
        Default constructor

        Parameters
        ----------
        latent_hosp_admissions_rv : RandomVariable
            Latent process for the hospital admissions.
        latent_infections_rv : RandomVariable
            The infections latent process (passed to RtInfectionsRenewalModel).
        gen_int_rv : RandomVariable
            Generation time (passed to RtInfectionsRenewalModel)
        I0_rv : RandomVariable
            Initial infections (passed to RtInfectionsRenewalModel)
        Rt_process_rv : RandomVariable
            Rt process  (passed to RtInfectionsRenewalModel).
        hosp_admission_obs_process_rv : RandomVariable, optional
            Observation process for the hospital admissions.

        Returns
        -------
        None
        """
        self.basic_renewal = RtInfectionsRenewalModel(
            gen_int_rv=gen_int_rv,
            I0_rv=I0_rv,
            latent_infections_rv=latent_infections_rv,
            infection_obs_process_rv=None,  # why is this None?
            Rt_process_rv=Rt_process_rv,
        )

        HospitalAdmissionsModel.validate(
            latent_hosp_admissions_rv, hosp_admission_obs_process_rv
        )

        self.latent_hosp_admissions_rv = latent_hosp_admissions_rv
        if hosp_admission_obs_process_rv is None:
            hosp_admission_obs_process_rv = NullObservation()

        self.hosp_admission_obs_process_rv = hosp_admission_obs_process_rv

    @staticmethod
    def validate(
        latent_hosp_admissions_rv, hosp_admission_obs_process_rv
    ) -> None:
        """
        Verifies types and status (RV) of latent and observed hospital admissions

        Parameters
        ----------
        latent_hosp_admissions_rv : RandomVariable
            The latent process for the hospital admissions.
        hosp_admission_obs_process_rv : RandomVariable
            The observed hospital admissions.

        Returns
        -------
        None
        """
        assert isinstance(latent_hosp_admissions_rv, RandomVariable)
        if hosp_admission_obs_process_rv is not None:
            assert isinstance(hosp_admission_obs_process_rv, RandomVariable)

        return None

    def sample(
        self,
        n_datapoints: int | None = None,
        data_observed_hosp_admissions: ArrayLike | None = None,
        padding: int = 0,
        **kwargs,
    ) -> HospModelSample:
        """
        Sample from the HospitalAdmissions model

        Parameters
        ----------
        n_datapoints : int, optional
            Number of timepoints to sample (passed to the basic renewal model).
        data_observed_hosp_admissions : ArrayLike, optional
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
        sample_observed_admissions : For sampling observed hospital admissions
        """
        if n_datapoints is None and data_observed_hosp_admissions is None:
            raise ValueError(
                "Either n_datapoints or data_observed_hosp_admissions "
                "must be passed."
            )
        elif (
            n_datapoints is not None
            and data_observed_hosp_admissions is not None
        ):
            raise ValueError(
                "Cannot pass both n_datapoints and data_observed_hosp_admissions."
            )
        elif n_datapoints is None:
            n_datapoints = len(data_observed_hosp_admissions)
        else:
            n_datapoints = n_datapoints

        # Getting the initial quantities from the basic model
        basic_model = self.basic_renewal.sample(
            n_datapoints=n_datapoints,
            data_observed_infections=None,
            padding=padding,
            **kwargs,
        )

        # Sampling the latent hospital admissions
        (
            infection_hosp_rate,
            latent_hosp_admissions,
            *_,
        ) = self.latent_hosp_admissions_rv(
            latent_infections=basic_model.latent_infections,
            **kwargs,
        )

        (
            observed_hosp_admissions,
            *_,
        ) = self.hosp_admission_obs_process_rv(
            mu=latent_hosp_admissions[-n_datapoints:],
            obs=data_observed_hosp_admissions,
            **kwargs,
        )

        return HospModelSample(
            Rt=basic_model.Rt,
            latent_infections=basic_model.latent_infections,
            infection_hosp_rate=infection_hosp_rate,
            latent_hosp_admissions=latent_hosp_admissions,
            observed_hosp_admissions=observed_hosp_admissions,
        )
