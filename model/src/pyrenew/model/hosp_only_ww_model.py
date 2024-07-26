# numpydoc ignore=GL08
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


class hosp_only_ww_model(Model):  # numpydoc ignore=GL08
    def __init__(
        self,
        state_pop,
        i0_over_n_rv,
        initialization_rate_rv,
        n_initialization_points,
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
        return None

    def validate(self):  # numpydoc ignore=GL08
        return None

    def sample(self):  # numpydoc ignore=GL08
        i0 = self.infection_initialization_process()
        log_r_mu_intercept_rv = DistributionalRV()
        log_r_mu_intercept_rv()
        # log_r_mu_intercept ~ normal(r_logmean, r_logsd)

        return i0
