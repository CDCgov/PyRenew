# numpydoc ignore=GL08
import numpyro.distributions as dist
import pyrenew.transformation as transformation
from pyrenew.latent import (
    InfectionInitializationProcess,
    InitializeInfectionsExponentialGrowth,
)
from pyrenew.metaclass import (
    DistributionalRV,
    Model,
    TransformedRandomVariable,
)
from pyrenew.process import FirstDifferenceARProcess


class hosp_only_ww_model(Model):  # numpydoc ignore=GL08
    def __init__(
        self,
        state_pop,
        i0_over_n_rv,
        initialization_rate_rv,
        log_r_mu_intercept_rv,
        autoreg_rt_rv,  # ar process
        eta_sd_rv,  # sd of random walk for ar process
        n_initialization_points,
        n_timepoints,
    ):  # numpydoc ignore=GL08
        self.infection_initialization_process = InfectionInitializationProcess(
            "I0_initialization",
            TransformedRandomVariable(
                "i0",
                i0_over_n_rv,
                transforms=transformation.AffineTransform(
                    loc=0, scale=state_pop
                ),
            ),
            InitializeInfectionsExponentialGrowth(
                n_initialization_points,
                initialization_rate_rv,
                t_pre_init=0,
            ),
            t_unit=1,
        )

        self.autoreg_rt_rv = autoreg_rt_rv
        self.eta_sd_rv = eta_sd_rv
        self.log_r_mu_intercept_rv = log_r_mu_intercept_rv
        self.n_timepoints = n_timepoints
        return None

    def validate(self):  # numpydoc ignore=GL08
        return None

    def sample(self):  # numpydoc ignore=GL08
        i0 = self.infection_initialization_process()

        autoreg_rt = self.autoreg_rt_rv()[0].value
        print(autoreg_rt)
        eta_sd = self.eta_sd_rv()[0].value
        my_fd_ar = FirstDifferenceARProcess(
            "first_diff_ar", autoreg_rt, eta_sd
        )
        init_rate_of_change_rv = DistributionalRV(
            "init_rate_of_change", dist.Normal(0, eta_sd)
        )
        init_rate_of_change = init_rate_of_change_rv()
        log_r_mu_intercept = self.log_r_mu_intercept_rv()
        fd_sample = my_fd_ar.sample(
            self.n_timepoints, log_r_mu_intercept, init_rate_of_change
        )
        # not sure about the length
        # gotta exponentiate somewhere, maybe with transformed random variable?
        # log_r_mu_intercept_rv()
        # log_r_mu_intercept ~ normal(r_logmean, r_logsd)

        return (i0, fd_sample)
