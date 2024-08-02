# numpydoc ignore=GL08
import jax.numpy as jnp
import numpyro.distributions as dist
import pyrenew.transformation as transformation
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import (
    InfectionInitializationProcess,
    InfectionsWithFeedback,
    InitializeInfectionsExponentialGrowth,
)
from pyrenew.metaclass import (  # TransformedRandomVariable,
    DistributionalRV,
    Model,
)
from pyrenew.process import ARProcess, RtWeeklyDiffProcess


class hosp_only_ww_model(Model):  # numpydoc ignore=GL08
    def __init__(
        self,
        state_pop,
        i0_over_n_rv,
        initialization_rate_rv,
        log_r_mu_intercept_rv,
        autoreg_rt_rv,  # ar process
        eta_sd_rv,  # sd of random walk for ar process
        generation_interval_pmf_rv,
        infection_feedback_strength_rv,
        infection_feedback_pmf_rv,
        p_hosp_mean_rv,
        p_hosp_w_sd_rv,
        autoreg_p_hosp_rv,
        hosp_wday_effect_rv,
        inf_to_hosp_rv,
        n_initialization_points,
    ):  # numpydoc ignore=GL08
        self.infection_initialization_process = InfectionInitializationProcess(
            "I0_initialization",
            i0_over_n_rv,
            InitializeInfectionsExponentialGrowth(
                n_initialization_points,
                initialization_rate_rv,
                t_pre_init=-50,
            ),
            t_unit=1,
        )

        self.inf_with_feedback_proc = InfectionsWithFeedback(
            infection_feedback_strength=infection_feedback_strength_rv,
            infection_feedback_pmf=infection_feedback_pmf_rv,
        )

        self.autoreg_rt_rv = autoreg_rt_rv
        self.eta_sd_rv = eta_sd_rv
        self.log_r_mu_intercept_rv = log_r_mu_intercept_rv
        self.generation_interval_pmf_rv = generation_interval_pmf_rv
        self.infection_feedback_pmf_rv = infection_feedback_pmf_rv
        self.p_hosp_mean_rv = p_hosp_mean_rv
        self.p_hosp_w_sd_rv = p_hosp_w_sd_rv
        self.autoreg_p_hosp_rv = autoreg_p_hosp_rv
        self.hosp_wday_effect_rv = hosp_wday_effect_rv
        self.inf_to_hosp_rv = inf_to_hosp_rv
        self.state_pop = state_pop
        return None

    def validate(self):  # numpydoc ignore=GL08
        return None

    def sample(self, n_timepoints):  # numpydoc ignore=GL08
        n_weeks = n_timepoints // 7 + 1

        i0 = self.infection_initialization_process()

        eta_sd = self.eta_sd_rv()[0].value

        autoreg_rt = self.autoreg_rt_rv()[0].value
        autoreg_rt_det_rv = DeterministicVariable("autoreg_rt_det", autoreg_rt)

        log_r_mu_intercept = self.log_r_mu_intercept_rv()[0].value

        init_rate_of_change_rv = DistributionalRV(
            "init_rate_of_change",
            dist.Normal(0, eta_sd / jnp.sqrt(1 - jnp.pow(autoreg_rt, 2))),
        )

        init_rate_of_change = init_rate_of_change_rv()[0].value

        rt_proc = RtWeeklyDiffProcess(
            name="rtu_weekly_diff",
            offset=0,
            log_rt_rv=DeterministicVariable(
                name="log_rt",
                value=jnp.array(
                    [
                        log_r_mu_intercept,
                        log_r_mu_intercept + init_rate_of_change,
                    ]
                ),
            ),
            autoreg_rv=autoreg_rt_det_rv,
            periodic_diff_sd_rv=DeterministicVariable(
                name="periodic_diff_sd", value=jnp.array(eta_sd)
            ),
        )

        rtu = rt_proc.sample(duration=n_timepoints)
        generation_interval_pmf = self.generation_interval_pmf_rv()

        inf_with_feedback_proc_sample = self.inf_with_feedback_proc.sample(
            Rt=rtu[0].value,
            I0=i0[0].value,
            gen_int=generation_interval_pmf[0].value,
        )

        latent_infections = jnp.concat(
            [
                i0[0].value,
                inf_with_feedback_proc_sample.post_initialization_infections.value,
            ]
        )

        p_hosp_mean = self.p_hosp_mean_rv()[0].value
        p_hosp_w_sd = self.p_hosp_w_sd_rv()[0].value
        autoreg_p_hosp = self.autoreg_p_hosp_rv()[0].value

        p_hosp_ar_proc = ARProcess(
            "p_hosp",
            mean=p_hosp_mean,
            autoreg=autoreg_p_hosp,
            noise_sd=p_hosp_w_sd,
        )

        p_hosp_ar_init_rv = DistributionalRV(
            "p_hosp_ar_init",
            dist.Normal(
                p_hosp_mean,
                p_hosp_w_sd / jnp.sqrt(1 - jnp.pow(autoreg_p_hosp, 2)),
            ),
        )
        p_hosp_ar_init = p_hosp_ar_init_rv()[0].value

        p_hosp_ar = p_hosp_ar_proc.sample(
            duration=n_weeks, inits=p_hosp_ar_init
        )

        ihr = jnp.repeat(
            transformation.SigmoidTransform()(p_hosp_ar[0].value), repeats=7
        )[
            :n_timepoints
        ]  # this is only applied after the hospital_admissions are generated, not to all the latent infectios

        hosp_wday_effect_raw = self.hosp_wday_effect_rv()[0].value
        hosp_wday_effect = jnp.tile(hosp_wday_effect_raw, n_weeks)[
            :n_timepoints
        ]
        inf_to_hosp = self.inf_to_hosp_rv()[0].value

        potential_latent_hospital_admissions = jnp.convolve(
            latent_infections,
            inf_to_hosp,
            mode="full",
        )[:n_timepoints]

        latent_hospital_admissions = (
            potential_latent_hospital_admissions * ihr * hosp_wday_effect
        )

        return (
            i0,
            rtu,
            inf_with_feedback_proc_sample,
            p_hosp_ar,
            ihr,
            hosp_wday_effect,
            potential_latent_hospital_admissions,
            latent_hospital_admissions,
        )
