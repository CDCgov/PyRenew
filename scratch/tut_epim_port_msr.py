"""
Porting `cfa-forecast-renewal-epidemia`
over to MSR. Validated on NHSN influenza
data from the 2023-24 season.
"""

import datetime as dt
import os
from pprint import pprint

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
import pyrenew.transformation as t
import toml
from jax.typing import ArrayLike
from matplotlib import font_manager as fm
from numpyro.infer.reparam import LocScaleReparam
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import (
    InfectionInitializationProcess,
    InitializeInfectionsExponentialGrowth,
    compute_infections_from_rt,
    logistic_susceptibility_adjustment,
)
from pyrenew.metaclass import (
    DistributionalRV,
    Model,
    RandomVariable,
    TransformedRandomVariable,
)
from pyrenew.observation import NegativeBinomialObservation
from pyrenew.process import SimpleRandomWalkProcess
from pyrenew.regression import GLMPrediction

FONT_PATH = "texgyreschola-regular.otf"
TITLE_FONT_PROP = fm.FontProperties(fname=FONT_PATH, size=17.5)
LABEL_FONT_PROP = fm.FontProperties(fname=FONT_PATH, size=12.5)
AXES_FONT_PROP = fm.FontProperties(fname=FONT_PATH, size=16.5)

plt.style.use("epim_port.mplstyle")


class Config:  # numpydoc ignore=GL08
    def __init__(self, config_dict):  # numpydoc ignore=GL08
        for key, value in config_dict.items():
            setattr(self, key, value)


def load_influenza_hosp_data(
    data_path: str, print_first_n_rows: bool = False, n_row_count: int = 15
):
    """
    Loads NHSN hospitalization data from
    tsv file path.

    Parameters
    ----------
    data_path : str
        The path to the tsv file to be read.
    print_first_n_rows : bool
        Whether to print the data rows.
    n_row_count : int
        How many rows to print.

    Returns
    -------
    pl.DataFrame
        A polars dataframe of NHSN
        hospitalization data.
    """
    data = pl.read_csv(data_path, separator="\t", infer_schema_length=10000)
    # verification: columns
    print(data.columns)
    if print_first_n_rows and (5 <= n_row_count <= 50):
        pl.Config.set_tbl_hide_dataframe_shape(True)
        pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
        pl.Config.set_tbl_hide_column_data_types(True)
        with pl.Config(tbl_rows=n_row_count, tbl_cols=6):
            # verification: rows and columns of data
            print(data)
    return data


def plot_utils(
    axes: mpl.axes,
    figure: plt.figure,
    use_log: bool = False,
    title: str = "",
    ylabel: str = "",
    xlabel: str = "",
    use_legend: bool = False,
    display: bool = True,
    filename: str = "delete_me",
    save_as_img: bool = False,
    save_to_pdf: bool = False,
):  # numpydoc ignore=GL08
    if use_legend:
        axes.legend(loc="best")
    if use_log:
        axes.set_yscale("log")
        axes.set_ylabel(ylabel + " (Log-Scale)", fontproperties=AXES_FONT_PROP)
    axes.set_title(
        title,
        fontproperties=TITLE_FONT_PROP,
    )
    if not use_log:
        axes.set_xlabel(xlabel, fontproperties=AXES_FONT_PROP)
        axes.set_ylabel(ylabel, fontproperties=AXES_FONT_PROP)
    for label in axes.get_xticklabels():
        label.set_rotation(45)
        label.set_fontproperties(LABEL_FONT_PROP)
    for label in axes.get_yticklabels():
        label.set_fontproperties(LABEL_FONT_PROP)
    if display:
        plt.tight_layout()
        plt.show()
    if not os.path.exists(f"../figures/{filename}.png"):
        plt.tight_layout()
        if save_as_img:
            figure.savefig(f"../figures/{filename}")
        if save_to_pdf:
            return figure
    return None


def base_object_plot(
    y: np.ndarray,
    X: np.ndarray,
    title: str = "",
    X_label: str = "",
    Y_label: str = "",
    use_log: bool = False,
    use_legend: bool = False,
    display: bool = True,
    filename: str = "delete_me",
    save_as_img: bool = False,
    save_to_pdf: bool = False,
):  # numpydoc ignore=GL08
    figure, axes = plt.subplots(1, 1)
    axes.plot(X, y, color="black")
    plot_utils(
        figure=figure,
        axes=axes,
        title=title,
        xlabel=X_label,
        ylabel=Y_label,
        use_log=use_log,
        use_legend=use_legend,
        display=display,
        filename=filename,
        save_as_img=save_as_img,
        save_to_pdf=save_to_pdf,
    )


def plot_single_location_hosp_data(
    incidence_data: pl.DataFrame,
    states: str | list[str],
    lower_date: str,
    upper_date: str,
    use_log: bool = False,
    use_legend: bool = True,
    save_as_img: bool = False,
    save_to_pdf: bool = False,
    display: bool = True,
) -> None:
    """
    Plots incidence data between some
    lower and upper date (inclusive) for
    a single US territory.

    Parameters
    ----------
    incidence_data : pl.DataFrame
        A polars dataframe containing
        hospital admissions data.
    states : str | list[str]
        Two letter region abbreviation.
    lower_date : str
        Start date for data visualization.
    upper_date : str
        End date for data visualization.
    use_log : bool, optional
        Whether to use log-scaling on the
        y-axis. Defaults to False.
    use_legend : bool, optional
        Whether to use a legend. Defaults
        to True.
    save_as_img : bool, optional
        Whether to save the plot as an
        image. Defaults to False.
    save_to_pdf : bool, optional
        Whether to return the figure for
        use in a collected PDF of images.
    display : bool, optional
        Whether to show the image.

    Returns
    -------
    None | plt.figure
        Returns nothing if not saving
        to a pdf, otherwise return the
        figure.
    """
    # check states and set up linestyles
    if isinstance(states, str):
        states = [states]
    assert len(states) <= 4, "Use not more than four states."
    linestyles = ["solid", "dashed", "dotted", "dashdot"]
    # create figure and plot
    figure, axes = plt.subplots(1, 1)
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    axes.xaxis.set_major_locator(mdates.MonthLocator())
    # retrieve hospital admissions data and plot
    for i, state in enumerate(states):
        state_data = incidence_data.filter(
            (pl.col("location") == state)
            & (pl.col("date") >= lower_date)
            & (pl.col("date") <= upper_date)
        ).sort("date")
        observed_hosp_admissions = [
            int(elt) for elt in state_data["hosp"].to_list()
        ]
        dates = [
            dt.datetime.strptime(date, "%Y-%m-%d")
            for date in state_data["date"].to_list()
        ]
        axes.plot(
            dates,
            observed_hosp_admissions,
            color="black",
            linestyle=linestyles[i],
            label=state,
        )
    # other settings and saving figure
    ylabel = "Hospital Admissions"
    plot_utils(
        axes=axes,
        figure=figure,
        use_log=use_log,
        ylabel=ylabel,
        title=f"Reported Hospital Admissions: [{lower_date}, {upper_date}]",
        use_legend=use_legend,
        display=display,
        filename=f"{state}_{lower_date}-{upper_date}",
        save_as_img=save_as_img,
        save_to_pdf=save_to_pdf,
    )
    return None


class CFAEPIM_Infections(RandomVariable):  # numpydoc ignore=GL08
    @staticmethod
    def validate() -> None:  # numpydoc ignore=GL08
        return None

    def sample(
        self, Rt: ArrayLike, gen_int: ArrayLike, P: float, **kwargs
    ) -> tuple:  # numpydoc ignore=GL08
        I0_samples = self.I0.sample()
        I0 = I0_samples[0].value
        if I0.size < gen_int.size:
            raise ValueError(
                "Initial infections vector must be at least as long as "
                "the generation interval. "
                f"Initial infections vector length: {I0.size}, "
                f"generation interval length: {gen_int.size}."
            )
        gen_int_rev = jnp.flip(gen_int)
        recent_I0 = I0[-gen_int_rev.size :]
        all_infections = compute_infections_from_rt(
            I0=recent_I0,
            Rt=Rt,
            reversed_generation_interval_pmf=gen_int_rev,
        )
        S_t = jnp.zeros_like(all_infections)
        S_t = S_t.at[0].set(P)  # initial P

        # update: avoid set as much as possible
        # update: hstack(); can be changed uniformly later
        # update: per DB's update, use numpyro.contrib.flow something scan
        def update_infections(carry, x):  # numpydoc ignore=GL08
            S_prev, I_prev = carry
            Rt, gen_int_rev_t = x
            # update: ^ not actually the backwards looking convolve desired
            # verify this; use fixed value for gen_int doesn't need to change;
            # could work if you want a time varying generation interval;
            i_raw_t = Rt * jnp.dot(I_prev, gen_int_rev_t)
            i_t = logistic_susceptibility_adjustment(i_raw_t, S_prev / P, P)
            S_t = S_prev - i_t
            I_prev = jnp.roll(I_prev, -1)
            I_prev = I_prev.at[-1].set(i_t)
            return (S_t, I_prev), i_t

        # confirm: update this to set prior on S_{v-1} / P
        # update: the prior will change init_carry, [0, P]
        init_carry = (P, recent_I0)
        Rt_gen_int_rev = jnp.stack(
            [Rt, jnp.tile(gen_int_rev, (Rt.size, 1))], axis=-1
        )
        (_, all_S_t), all_I_t = jax.lax.scan(
            update_infections, init_carry, Rt_gen_int_rev
        )
        # update: realized Rt is a consequence of the sus. adjustment
        # Epidemia does not document this well.
        return all_I_t, all_S_t


class CFAEPIM_Rt(RandomVariable):  # numpydoc ignore=GL08
    def __init__(
        self, intercept_RW_prior, max_rt, gamma_RW_prior, gamma_RW_prior_scale
    ):  # numpydoc ignore=GL08
        self.intercept_RW_prior = intercept_RW_prior
        self.max_rt = max_rt
        self.gamma_RW_prior = gamma_RW_prior
        self.gamma_RW_prior_scale = gamma_RW_prior_scale

    @staticmethod
    def validate() -> None:  # numpydoc ignore=GL08
        pass

    def sample(self, n_steps: int, **kwargs) -> tuple:  # numpydoc ignore=GL08
        glm = GLMPrediction(
            name="CFAEPIM_Rt_GLM",
            fixed_predictor_values=jnp.ones((n_steps, 1)),  # intercept & RW
            intercept_prior=self.intercept_RW_prior,
            coefficient_priors=self.gamma_RW_prior,
            transform=t.ScaledLogitTransform(x_max=self.max_rt),
        )
        # update: Epidemia does this all as GLM under the hood,
        # but epimlight model just has Rt intercept & random walk
        # update: intercept can just be part of RtWalk, simple random
        # walk can be transformed using scaled logit, init_rv = intercept
        # prior, sd_wt (as you've done). These needs to be broadcast from
        # weekly to daily.
        eta_samples = glm.sample()
        sd_wt = numpyro.sample("Wt_rw_sd", dist.gamma_RW_prior_scale)
        wt_rv = SimpleRandomWalkProcess(
            name="Wt",
            step_rv=DistributionalRV(
                name="rw_step_rv",
                dist=dist.Normal(0, sd_wt),  # confirm: should I reuse
                reparam=LocScaleReparam(0),  # confirm: thoughts on this?
            ),
            init_rv=DistributionalRV(
                name="init_Wt_rv",
                dist=dist.Normal(
                    0, 1
                ),  # confirm: not sure what else to put here
            ),
        )
        wt_samples = wt_rv.sample(n_steps=n_steps, **kwargs)
        rt_samples = eta_samples["prediction"] + wt_samples[0].value
        # update: don't want scaled logit twice, remove GLM section;
        transformed_rt_samples = t.ScaledLogitTransform(x_max=self.max_rt)(
            rt_samples
        )
        return transformed_rt_samples

    # update: broadcast weekly to daily


class CFAEPIM_Model(Model):  # numpydoc ignore=GL08,PR01
    """
    The `cfaepim` model. Its properties are derived
    from the configuration file provided. Forecasts
    for a single state can be made using the model.
    The fundamental question, for each part of
    `cfaepim`, comes in asking whether a component
    ought to made in base MSR or made custom in MSR.
    Remember some parameters, e.g. first_fitting_date,
    are already accounted for in the data that has
    been made available.
    """

    def __init__(
        self, state: str, dataset: pl.DataFrame, config: any
    ):  # numpydoc ignore=GL08
        self.state = state
        self.dataset = dataset

        # population
        self.population = int(
            self.dataset.filter(pl.col("location") == state)
            .select(["population"])
            .unique()
            .to_numpy()[0][0]
        )

        # convert config values into instance attributes
        self.config = config
        for key, value in config.items():
            setattr(self, key, value)

        # reported admissions in proper format
        self.data_observed_hosp_admissions = np.array(
            self.dataset.filter(pl.col("location") == state).select(
                ["date", "hosp"]
            )["hosp"]
        )

        # predictors
        day_of_week_covariate = self.dataset.select(
            pl.col("day_of_week")
        ).to_dummies()
        remaining_covariates = self.dataset.select(
            ["is_holiday", "is_post_holiday"]
        )
        covariates = pl.concat(
            [day_of_week_covariate, remaining_covariates], how="horizontal"
        )
        self.predictors = covariates.to_numpy()

        # transmission: generation time distribution
        self.pmf_array = jnp.array(self.generation_time_dist)
        self.gen_int = DeterministicPMF(name="gen_int", value=self.pmf_array)

        # transmission: prior for RW intercept
        self.intercept_RW_prior = dist.Normal(
            self.rt_intercept_prior_mode, self.rt_intercept_prior_scale
        )

        # transmission: prior for gamma term
        self.gamma_RW_prior_scale = dist.HalfNormal(
            self.weekly_rw_prior_scale
        ).sample(jax.random.PRNGKey(self.seed))
        self.gamma_RW_prior = dist.Normal(0, self.gamma_RW_prior_scale)

        # transmission: Rt process, involving scaled-logit
        # link function g transforming a linear predictor for the
        # random walk
        self.Rt_process = CFAEPIM_Rt(
            intercept_RW_prior=self.intercept_RW_prior,
            max_rt=self.max_rt,
            gamma_RW_prior=self.gamma_RW_prior,
            gamma_RW_prior_scale=self.gamma_RW_prior_scale,
        )

        # infections:  initialized infections
        self.I0 = InfectionInitializationProcess(
            "I0_initialization",
            DistributionalRV(
                name="I0",
                dist=dist.LogNormal(loc=jnp.log(100), scale=jnp.log(1.75)),
            ),  # confirm: need to change from tutoral values
            InitializeInfectionsExponentialGrowth(
                n_timepoints=self.n_pre_observation_days,
                rate=DeterministicVariable(name="rate", value=0.05),
            ),  # update: initialize infection vector w/ exp DISTRIBUTED vector
            t_unit=1,
        )

        # initialize alpha_t and observation components
        self._init_alpha_t()
        self._init_observation_component()

        # infections component
        self.infections = CFAEPIM_Infections(self.I0)

    @staticmethod
    def validate() -> None:  # numpydoc ignore=GL08
        pass

    def _init_alpha_t(self):  # numpydoc ignore=GL08
        predictor_values = self.predictors
        alpha_intercept_prior = dist.Normal(
            self.ihr_intercept_prior_mode, self.ihr_intercept_prior_scale
        )
        day_of_week_priors = dist.Normal(0, 0.25).expand([6])
        holiday_prior = dist.Normal(
            self.holiday_eff_prior_mode, self.holiday_eff_prior_scale
        ).expand([1])
        post_holiday_prior = dist.Normal(
            self.post_holiday_eff_prior_mode, self.post_holiday_eff_prior_scale
        ).expand([1])
        pre_observation_prior = dist.Normal(
            self.non_obs_effect_prior_mode, self.non_obs_effect_prior_mode
        ).expand([1])
        all_coefficient_priors = jnp.concatenate(
            [
                day_of_week_priors,
                holiday_prior,
                post_holiday_prior,
                pre_observation_prior,
            ]
        )
        self.alpha_process = GLMPrediction(
            name="alpha_t",
            fixed_predictor_values=predictor_values,
            intercept_prior=alpha_intercept_prior,
            coefficient_priors=all_coefficient_priors,
            transform=t.ScaledLogitTransform(x_max=self.max_rt),
        )  # update: make this into its own class

    def _init_observation_component(self):  # numpydoc ignore=GL08
        # update: connect w/ alpha
        # update: combine w/ alpha as a class
        # update this; remove extraneous; rename; add reciprocal dispersion
        nb_conc_rv = TransformedRandomVariable(
            "concentration",
            base_rv=DistributionalRV(
                name="concentration_raw", dist=dist.Normal(0, 1)
            ),
            transforms=t.ScaledLogitTransform(x_max=self.max_rt),
        )
        self.admissions = NegativeBinomialObservation(
            name="hospital_admissions", concentration_rv=nb_conc_rv
        )

    def sample(self, **kwargs) -> tuple:  # numpydoc ignore=GL08
        pass

        # update: this needs to be update to reflection the
        # updated state of the components as classes

        # update: should never have...

        # n_steps = kwargs.get("n_steps", 100)
        # P = self.population
        # rt_samples = self.rt_process.sample(n_steps=n_steps)
        # alpha_samples = self.alpha_process.sample()
        # I0_samples = self.I0_rv.sample()
        # all_I_t, all_S_t = self.infections.sample(
        #     Rt=rt_samples,
        #     I0=I0_samples,
        #     gen_int=self.pmf_array,
        #     P=P

        # get rt_samples, ,

    def run(self, **kwargs):  # numpydoc ignore=GL08
        kernel = numpyro.infer.NUTS(
            model=self.model,
            step_size=self.adapt_delta,
            max_tree_depth=self.max_treedepth,
        )
        mcmc = numpyro.infer.MCMC(
            sampler=kernel,
            num_warmup=self.n_warmup,
            num_samples=self.n_iter,
            num_chains=self.n_chains,
        )
        mcmc.run(
            rng_key=jax.random.PRNGKey(self.seed),
            data_observed_hosp_admissions=self.data_observed_hosp_admissions,
        )
        mcmc.print_summary()
        self.mcmc = mcmc
        # update: defer to metaclass for model + run
        # update: represent data_observed_hosp_admissions as an attribute,
        # but not as an instance attributes; pass this to the.
        # "a model should have as few numbers as it in possible"
        # "any numbers that the model sees should be passed in at run time"
        # "config should be seen at run time"

    def predict(self, rng_key, **kwargs):  # numpydoc ignore=GL08
        posterior_pred_dist = numpyro.infer.Predictive(
            self.model, self.mcmc.get_samples()
        )
        predictions = posterior_pred_dist(
            rng_key=jax.random.PRNGKey(
                self.seed
            ),  # confirm: do not just define as instance attribute?
            num_samples=self.n_iter,
        )
        return predictions


def verify_cfaepim_MSR(cfaepim_MSR_model) -> None:  # numpydoc ignore=GL08
    # verification: population
    print(cfaepim_MSR_model.population)
    # verification: predictors
    print(cfaepim_MSR_model.predictors)
    # verification: Rt process
    print(cfaepim_MSR_model.Rt_process)
    # verification: generation interval deterministic PMF
    cfaepim_MSR_model.gen_int.validate(cfaepim_MSR_model.pmf_array)
    sampled_gen_int = cfaepim_MSR_model.gen_int.sample()
    print(f"Sampled Generation Interval: {sampled_gen_int}")
    base_object_plot(
        y=sampled_gen_int[0].value,
        X=np.arange(0, len(sampled_gen_int[0].value)),
        title="Sampled Generation Interval",
        filename="sample_generation_interval",
        save_as_img=True,
        display=False,
    )


def main():  # numpydoc ignore=GL08
    # determine number of CPU cores
    num_cores = os.cpu_count()
    numpyro.set_host_device_count(num_cores - (num_cores - 1))

    # load parameters config (2024-01-20)
    config = toml.load("./config/params_2024-01-20.toml")
    # verification: config file
    pprint(config)

    # load NHSN data w/ population counts (2024-01-20)
    data_path_01 = "../data/2024-01-20/2024-01-20_clean_data.tsv"
    influenza_hosp_data = load_influenza_hosp_data(
        data_path=data_path_01, print_first_n_rows=True, n_row_count=15
    )

    # verification: plot single state hospitalizations
    plot_single_location_hosp_data(
        incidence_data=influenza_hosp_data,
        states=["NY"],
        lower_date="2022-01-01",
        upper_date="2024-03-10",
        use_log=False,
        use_legend=True,
        save_as_img=True,
        save_to_pdf=False,
        display=False,
    )

    # update: move this content to the MSR repository,
    # merge or not merge, won't pollute repo. unless
    # someone permits it to

    # instantiate cfaepim-MSR
    cfaepim_MSR = CFAEPIM_Model(
        state="NY", dataset=influenza_hosp_data, config=config
    )

    # verify and visualize aspects of the model
    verify_cfaepim_MSR(cfaepim_MSR)

    # cfaepim_MSR.run(
    #     mcmc_args=dict(progress_bar=False, num_chains=2)
    # )

    # instantiate MSR-cfaepim model
    # simulate data
    # run the model for NY
    # print summary (print_summary)
    # visualize prior predictive (prior_predictive)
    # visualize posterior predictive (posterior_predictive)
    # spread draws (spread_draws)


if __name__ == "__main__":
    main()
