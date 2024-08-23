"""
Porting `cfa-forecast-renewal-epidemia`
over to MSR. Validated on NHSN influenza
data from the 2023-24 season.
"""
import argparse
import datetime as dt
import glob

# import inspect
import logging
import os
from datetime import datetime, timedelta
from typing import NamedTuple

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl

# import rpy2.robjects as ro
import toml
from jax.typing import ArrayLike
from matplotlib import font_manager as fm
from numpyro import render_model
from numpyro.infer.reparam import LocScaleReparam

import pyrenew.transformation as t
from pyrenew.deterministic import DeterministicPMF
from pyrenew.latent import (
    InfectionInitializationProcess,
    InitializeInfectionsFromVec,
    logistic_susceptibility_adjustment,
)
from pyrenew.metaclass import (
    DistributionalRV,
    Model,
    RandomVariable,
    SampledValue,
    TransformedRandomVariable,
)
from pyrenew.observation import NegativeBinomialObservation
from pyrenew.process import SimpleRandomWalkProcess
from pyrenew.regression import GLMPrediction

# from rpy2.robjects import pandas2ri
# from rpy2.robjects.packages import importr

FONT_PATH = "texgyreschola-regular.otf"
if os.path.exists(FONT_PATH):
    TITLE_FONT_PROP = fm.FontProperties(fname=FONT_PATH, size=17.5)
    LABEL_FONT_PROP = fm.FontProperties(fname=FONT_PATH, size=12.5)
    AXES_FONT_PROP = fm.FontProperties(fname=FONT_PATH, size=16.5)
if not os.path.exists(FONT_PATH):
    TITLE_FONT_PROP, LABEL_FONT_PROP, AXES_FONT_PROP = None, None, None

# use first mplstyle file that appears, should one exist
if len(glob.glob("*mplstyle")) != 0:
    plt.style.use(glob.glob("*mplstyle")[0])

CURRENT_DATE = datetime.today().strftime("%Y-%m-%d")
CURRENT_DATE_EXTENDED = datetime.today().strftime("%Y-%m-%d_%H:%M:%S")

HOLIDAYS = ["2023-11-23", "2023-12-25", "2023-12-31", "2024-01-01"]

JURISDICTIONS = [
    "AK",
    "AL",
    "AR",
    "AZ",
    "CA",
    "CO",
    "CT",
    "DC",
    "DE",
    "FL",
    "GA",
    "HI",
    "IA",
    "ID",
    "IL",
    "IN",
    "KS",
    "KY",
    "LA",
    "MA",
    "MD",
    "ME",
    "MI",
    "MN",
    "MO",
    "MS",
    "MT",
    "NC",
    "ND",
    "NE",
    "NH",
    "NJ",
    "NM",
    "NV",
    "NY",
    "OH",
    "OK",
    "OR",
    "PA",
    "PR",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "US",
    "UT",
    "VA",
    "VI",
    "VT",
    "WA",
    "WI",
    "WV",
    "WY",
]

# set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def display_data(
    data: pl.DataFrame,
    n_row_count: int = 15,
    n_col_count: int = 5,
    first_only: bool = False,
    last_only: bool = False,
) -> None:
    """
    Display the columns and rows of
    a polars dataframe.

    Parameters
    ----------
    data : pl.DataFrame
        A polars dataframe.
    n_row_count : int, optional
        How many rows to print.
        Defaults to 15.
    n_col_count : int, optional
        How many columns to print.
        Defaults to 15.
    first_only : bool, optional
        If True, only display the first `n_row_count` rows. Defaults to False.
    last_only : bool, optional
        If True, only display the last `n_row_count` rows. Defaults to False.

    Returns
    -------
    None
        Displays data.
    """
    rows, cols = data.shape
    assert (
        1 <= n_col_count <= cols
    ), f"Must have reasonable column count; was type {n_col_count}"
    assert (
        1 <= n_row_count <= rows
    ), f"Must have reasonable row count; was type {n_row_count}"
    assert (
        first_only + last_only
    ) != 2, "Can only do one of last or first only."
    if first_only:
        data_to_display = data.head(n_row_count)
    elif last_only:
        data_to_display = data.tail(n_row_count)
    else:
        data_to_display = data.head(n_row_count)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_hide_column_data_types(True)
    with pl.Config(tbl_rows=n_row_count, tbl_cols=n_col_count):
        print(f"Dataset In Use For `cfaepim`:\n{data_to_display}\n")


def check_file_path_valid(file_path: str) -> None:
    """
    Checks if a file path is valid. Used to check
    the entered data and config paths.

    Parameters
    ----------
    file_path : str
        Path to the file (usually data or config).

    Returns
    -------
    None
        Checks files.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The path '{file_path}' does not exist.")
    if not os.path.isfile(file_path):
        raise IsADirectoryError(f"The path '{file_path}' is not a file.")
    return None


def load_data(
    data_path: str,
    sep: str = "\t",
    schema_length: int = 10000,
) -> pl.DataFrame:
    """
    Loads historical (i.e., `.tsv` data generated
    `cfaepim` for a weekly run) data.

    Parameters
    ----------
    data_path : str
        The path to the tsv file to be read.
    sep : str, optional
        The separator between values in the
        data file. Defaults to tab-separated.
    schema_length : int, optional
        An approximation of the expected
        maximum number of rows. Defaults
        to 10000.

    Returns
    -------
    pl.DataFrame
        An unvetted polars dataframe of NHSN
        hospitalization data.
    """
    check_file_path_valid(file_path=data_path)
    assert sep in [
        "\t",
        ",",
    ], f"Separator must be tabs or commas; was type {sep}"
    assert (
        7500 <= schema_length <= 25000
    ), f"Schema length must be reasonable; was type {schema_length}"
    data = pl.read_csv(
        data_path, separator=sep, infer_schema_length=schema_length
    )
    return data


def load_config(config_path: str) -> dict[str, any]:
    """
    Attempts to load config toml file.

    Parameters
    ----------
    config_path : str
        The path to the configuration file,
        read in via argparse.

    Returns
    -------
    config
        A dictionary of variables with
        associates values for `cfaepim`.
    """
    check_file_path_valid(file_path=config_path)
    try:
        config = toml.load(config_path)
    except toml.TomlDecodeError as e:
        raise ValueError(
            f"Failed to parse the TOML file at '{config_path}'; error: {e}"
        )
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred while reading the TOML file: {e}"
        )
    return config


def ensure_output_directory(args: dict[str, any]):  # numpydoc ignore=GL08
    output_directory = "./output/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if args.historical_data:
        output_directory += f"Historical_{args.reporting_date}/"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    if not args.historical_data:
        output_directory += f"{args.reporting_date}/"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    return output_directory


def assert_historical_data_files_exist(
    reporting_date: str,
):  # numpydoc ignore=GL08
    data_directory = f"./data/{reporting_date}/"
    assert os.path.exists(
        data_directory
    ), f"Data directory {data_directory} does not exist."
    required_files = [
        f"{reporting_date}_clean_data.tsv",
        f"{reporting_date}_config.toml",
        f"{reporting_date}-cfarenewal-cfaepimlight.csv",
    ]
    for file in required_files:
        assert os.path.exists(
            os.path.join(data_directory, file)
        ), f"Required file {file} does not exist in {data_directory}."
    return data_directory


def plot_utils(
    axes: mpl.axes.Axes,
    figure: plt.Figure,
    use_log: bool = False,
    title: str = "",
    ylabel: str = "",
    xlabel: str = "",
    use_legend: bool = True,
    display: bool = True,
    filename: str = "delete_me",
    save_as_img: bool = False,
    save_to_pdf: bool = False,
) -> None | plt.Figure:  # numpydoc ignore=GL08
    if use_legend:
        axes.legend(loc="best")
    if use_log:
        axes.set_yscale("log")
        axes.set_ylabel(ylabel + " (Log-Scale)", fontproperties=AXES_FONT_PROP)
    axes.set_title(title, fontproperties=TITLE_FONT_PROP)
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
    if save_as_img or save_to_pdf:
        plt.tight_layout()
        if save_as_img:
            if not os.path.exists("./figures"):
                os.makedirs("./figures")
            figure.savefig(f"./figures/{filename}.png")
        if save_to_pdf:
            return figure
    return None


def base_line_plot(
    y: list[np.ndarray],
    X: np.ndarray,
    labels: list[str] = [""],
    title: str = "",
    X_label: str = "",
    Y_label: str = "",
    use_log: bool = False,
    use_legend: bool = True,
    display: bool = True,
    filename: str = "delete_me",
    save_as_img: bool = False,
    save_to_pdf: bool = False,
) -> None | plt.Figure:  # numpydoc ignore=RT01
    """
    Simple X, y plot with title, labels, and
    some save features. Plot based on style
    defined in the style file. Defers non-plotting
    items (labeling, saving) to plot utils.
    """
    figure, axes = plt.subplots(1, 1)
    for i, vals in enumerate(y):
        axes.plot(X, vals, label=labels[i])
    axes.set_xlim(left=0)
    axes.set_ylim(bottom=0)
    figure = plot_utils(
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
    return figure


def base_mcmc_plot(
    y: list[np.ndarray],
    X: np.ndarray,
    labels: list[str] = [""],
    title: str = "",
    X_label: str = "",
    Y_label: str = "",
    use_log: bool = False,
    use_legend: bool = True,
    display: bool = True,
    filename: str = "delete_me",
    save_as_img: bool = False,
    save_to_pdf: bool = False,
) -> None | plt.Figure:  # numpydoc ignore=RT01
    """
    Simple plot for mcmc output with title, labels, and
    some save features. Plot based on style
    defined in the style file. Defers non-plotting
    items (labeling, saving) to plot utils.
    """
    figure, axes = plt.subplots(1, 1)
    for i, vals in enumerate(y):
        axes.plot(X, vals, label=labels[i])
    axes.set_xlim(left=0)
    axes.set_ylim(bottom=0)
    figure = plot_utils(
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
    return figure


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
) -> None | plt.Figure:
    """
    Plots incidence data between some lower and upper date
    (inclusive) for a single US territory.

    Parameters
    ----------
    incidence_data : pl.DataFrame
        A polars dataframe containing hospital admissions data.
    states : str | list[str]
        Two letter region abbreviation.
    lower_date : str
        Start date for data visualization.
    upper_date : str
        End date for data visualization.
    use_log : bool, optional
        Whether to use log-scaling on the y-axis. Defaults to False.
    use_legend : bool, optional
        Whether to use a legend. Defaults to True.
    save_as_img : bool, optional
        Whether to save the plot as an image. Defaults to False.
    save_to_pdf : bool, optional
        Whether to return the figure for use in a collected PDF of images.
    display : bool, optional
        Whether to show the image.

    Returns
    -------
    None | plt.Figure
        Returns nothing if not saving to a PDF, otherwise returns the figure.
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
    return plot_utils(
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


class CFAEPIM_Infections(RandomVariable):
    """
    Class representing the infection process in
    the CFAEPIM model. This class handles the sampling of
    infection counts over time, considering the
    reproduction number, generation interval, and population size,
    while accounting for susceptibility depletion.

    Parameters
    ----------
    I0 : ArrayLike
        Initial infection counts.
    susceptibility_prior : numpyro.distributions
        Prior distribution for the susceptibility proportion
        (S_{v-1} / P).
    """

    def __init__(
        self,
        I0: ArrayLike,
        susceptibility_prior: numpyro.distributions,
    ):  # numpydoc ignore=GL08
        logging.info("Initializing CFAEPIM_Infections")

        self.I0 = I0
        self.susceptibility_prior = susceptibility_prior

    @staticmethod
    def validate(I0: any, susceptibility_prior: any) -> None:
        """
        Validate the parameters of the
        infection process. Checks that the initial infections
        (I0) and susceptibility_prior are
        correctly specified. If any parameter is invalid,
        an appropriate error is raised.

        Raises
        ------
        TypeError
            If I0 is not array-like or
            susceptibility_prior is not
            a numpyro distribution.
        """
        logging.info("Validating CFAEPIM_Infections parameters")
        if not isinstance(I0, (np.ndarray, jnp.ndarray)):
            raise TypeError(
                f"Initial infections (I0) must be an array-like structure; was type {type(I0)}"
            )

        if not isinstance(susceptibility_prior, dist.Distribution):
            raise TypeError(
                f"susceptibility_prior must be a numpyro distribution; was type {type(susceptibility_prior)}"
            )

    def sample(
        self, Rt: ArrayLike, gen_int: ArrayLike, P: float, **kwargs
    ) -> tuple:
        """
        Given an array of reproduction numbers,
        a generation interval, and the size of a
        jurisdiction's population,
        calculate infections under the scheme
        of susceptible depletion.

        Parameters
        ----------
        Rt : ArrayLike
            Reproduction numbers over time; this is an array of
            Rt values for each time step.
        gen_int : ArrayLike
            Generation interval probability mass function. This is
            an array of probabilities representing the
            distribution of times between successive infections
            in a chain of transmission.
        P : float
            Population size. This is the total population
            size used for susceptibility adjustment.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample calls, should there be any.

        Returns
        -------
        tuple
            A tuple containing two arrays: all_I_t, an array of
            latent infections at each time step and all_S_t, an
            array of susceptible individuals at each time step.

        Raises
        ------
        ValueError
            If the length of the initial infections
            vector (I0) is less than the length of
            the generation interval.
        """

        # get initial infections
        I0_samples = self.I0.sample()
        I0 = I0_samples[0].value

        logging.debug(f"I0 samples: {I0}")

        # reverse generation interval (recency)
        gen_int_rev = jnp.flip(gen_int)

        if I0.size < gen_int.size:
            raise ValueError(
                "Initial infections vector must be at least as long as "
                "the generation interval. "
                f"Initial infections vector length: {I0.size}, "
                f"generation interval length: {gen_int.size}."
            )
        recent_I0 = I0[-gen_int_rev.size :]

        # sample the initial susceptible population proportion S_{v-1} / P from prior
        init_S_proportion = numpyro.sample(
            "S_v_minus_1_over_P", self.susceptibility_prior
        )
        logging.debug(f"Initial susceptible proportion: {init_S_proportion}")

        # calculate initial susceptible population S_{v-1}
        init_S = init_S_proportion * P

        def update_infections(carry, Rt):  # numpydoc ignore=GL08
            S_t, I_recent = carry

            # compute raw infections
            i_raw_t = Rt * jnp.dot(I_recent, gen_int_rev)

            # apply the logistic susceptibility adjustment to a potential new incidence
            i_t = logistic_susceptibility_adjustment(
                I_raw_t=i_raw_t, frac_susceptible=S_t / P, n_population=P
            )

            # update susceptible population
            S_t -= i_t

            # update infections
            I_recent = jnp.concatenate([I_recent[:-1], jnp.array([i_t])])

            return (S_t, I_recent), i_t

        # initial carry state
        init_carry = (init_S, recent_I0)

        # scan to iterate over time steps and update infections
        (all_S_t, _), all_I_t = numpyro.contrib.control_flow.scan(
            update_infections, init_carry, Rt
        )

        logging.debug(f"All infections: {all_I_t}")
        logging.debug(f"All susceptibles: {all_S_t}")

        return all_I_t, all_S_t


class CFAEPIM_Observation(RandomVariable):
    """
    Class representing the observation process
    in the CFAEPIM model. This class handles the generation
    of the alpha (instantaneous ascertaintment rate) process
    and the negative binomial observation process for
    modeling hospitalizations from latent infections.

    Parameters
    ----------
    predictors : ArrayLike
        Array of predictor (covariates) values for the alpha process.
    alpha_prior_dist : numpyro.distributions
        Prior distribution for the intercept in the alpha process.
    coefficient_priors : numpyro.distributions
        Prior distributions for the coefficients in the alpha process.
    nb_concentration_prior : numpyro.distributions
        Prior distribution for the concentration parameter of
        the negative binomial distribution.
    """

    def __init__(
        self,
        predictors,
        alpha_prior_dist,
        coefficient_priors,
        nb_concentration_prior,
    ):  # numpydoc ignore=GL08
        logging.info("Initializing CFAEPIM_Observation")

        CFAEPIM_Observation.validate(
            predictors,
            alpha_prior_dist,
            coefficient_priors,
            nb_concentration_prior,
        )

        self.predictors = predictors
        self.alpha_prior_dist = alpha_prior_dist
        self.coefficient_priors = coefficient_priors
        self.nb_concentration_prior = nb_concentration_prior

        self._init_alpha_t()
        self._init_negative_binomial()

    def _init_alpha_t(self):
        """
        Initialize the alpha process using a generalized
        linear model (GLM) (transformed linear predictor).
        The transform is set to the inverse of the sigmoid
        transformation.
        """
        logging.info("Initializing alpha process")
        self.alpha_process = GLMPrediction(
            name="alpha_t",
            fixed_predictor_values=self.predictors,
            intercept_prior=self.alpha_prior_dist,
            coefficient_priors=self.coefficient_priors,
            transform=t.SigmoidTransform().inv,
        )

    def _init_negative_binomial(self):
        """
        Sets up the negative binomial
        distribution for modeling hospitalizations
        with a prior on the concentration parameter.
        """
        logging.info("Initializing negative binomial process")
        self.nb_observation = NegativeBinomialObservation(
            name="negbinom_rv",
            concentration_rv=DistributionalRV(
                name="nb_concentration",
                dist=self.nb_concentration_prior,
            ),
        )

    @staticmethod
    def validate(
        predictors: any,
        alpha_prior_dist: any,
        coefficient_priors: any,
        nb_concentration_prior: any,
    ) -> None:
        """
        Validate the parameters of the CFAEPIM observation process. Checks that
        the predictors, alpha prior distribution, coefficient priors, and negative
        binomial concentration prior are correctly specified. If any parameter
        is invalid, an appropriate error is raised.
        """
        logging.info("Validating CFAEPIM_Observation parameters")
        if not isinstance(predictors, (np.ndarray, jnp.ndarray)):
            raise TypeError(
                f"Predictors must be an array-like structure; was type {type(predictors)}"
            )
        if not isinstance(alpha_prior_dist, dist.Distribution):
            raise TypeError(
                f"alpha_prior_dist must be a numpyro distribution; was type {type(alpha_prior_dist)}"
            )
        if not isinstance(coefficient_priors, dist.Distribution):
            raise TypeError(
                f"coefficient_priors must be a numpyro distribution; was type {type(coefficient_priors)}"
            )
        if not isinstance(nb_concentration_prior, dist.Distribution):
            raise TypeError(
                f"nb_concentration_prior must be a numpyro distribution; was type {type(nb_concentration_prior)}"
            )

    def sample(
        self,
        infections: ArrayLike,
        inf_to_hosp_dist: ArrayLike,
        **kwargs,
    ) -> tuple:
        """
        Sample from the observation process. Generates samples
        from the alpha process and calculates the expected number
        of hospitalizations by convolving the infections with
        the infection-to-hospitalization (delay distribution)
        distribution. It then samples from the negative binomial
        distribution to model the observed
        hospitalizations.

        Parameters
        ----------
        infections : ArrayLike
            Array of infection counts over time.
        inf_to_hosp_dist : ArrayLike
            Array representing the distribution of times
            from infection to hospitalization.
        **kwargs : dict, optional
            Additional keyword arguments passed through
            to internal sample calls, should there be any.

        Returns
        -------
        tuple
            A tuple containing the sampled instantaneous
            ascertainment values and the expected
            hospitalizations.
        """
        alpha_samples = self.alpha_process.sample()["prediction"]
        alpha_samples = alpha_samples[: infections.shape[0]]
        expected_hosp = (
            alpha_samples
            * jnp.convolve(infections, inf_to_hosp_dist, mode="full")[
                : infections.shape[0]
            ]
        )
        logging.debug(f"Alpha samples: {alpha_samples}")
        logging.debug(f"Expected hospitalizations: {expected_hosp}")
        return alpha_samples, expected_hosp


class CFAEPIM_Rt(RandomVariable):  # numpydoc ignore=GL08
    def __init__(
        self,
        intercept_RW_prior: numpyro.distributions,
        max_rt: float,
        gamma_RW_prior_scale: float,
        week_indices: ArrayLike,
    ):  # numpydoc ignore=GL08
        """
        Initialize the CFAEPIM_Rt class.

        Parameters
        ----------
        intercept_RW_prior : numpyro.distributions.Distribution
            Prior distribution for the random walk intercept.
        max_rt : float
            Maximum value of the reproduction number. Used as
            the scale in the `ScaledLogitTransform()`.
        gamma_RW_prior_scale : float
            Scale parameter for the HalfNormal distribution
            used for random walk standard deviation.
        week_indices : ArrayLike
            Array of week indices used for broadcasting
            the Rt values.
        """
        logging.info("Initializing CFAEPIM_Rt")
        self.intercept_RW_prior = intercept_RW_prior
        self.max_rt = max_rt
        self.gamma_RW_prior_scale = gamma_RW_prior_scale
        self.week_indices = week_indices

    @staticmethod
    def validate(
        intercept_RW_prior: any,
        max_rt: any,
        gamma_RW_prior_scale: any,
        week_indices: any,
    ) -> None:  # numpydoc ignore=GL08
        """
        Validate the parameters of the CFAEPIM_Rt class.

        Raises
        ------
        ValueError
            If any of the parameters are not valid.
        """
        logging.info("Validating CFAEPIM_Rt parameters")
        if not isinstance(intercept_RW_prior, dist.Distribution):
            raise ValueError(
                f"intercept_RW_prior must be a numpyro distribution; was type {type(intercept_RW_prior)}"
            )
        if not isinstance(max_rt, (float, int)) or max_rt <= 0:
            raise ValueError(
                f"max_rt must be a positive number; was type {type(max_rt)}"
            )
        if (
            not isinstance(gamma_RW_prior_scale, (float, int))
            or gamma_RW_prior_scale <= 0
        ):
            raise ValueError(
                f"gamma_RW_prior_scale must be a positive number; was type {type(gamma_RW_prior_scale)}"
            )
        if not isinstance(week_indices, (np.ndarray, jnp.ndarray)):
            raise ValueError(
                f"week_indices must be an array-like structure; was type {type(week_indices)}"
            )

    def sample(self, n_steps: int, **kwargs) -> tuple:  # numpydoc ignore=GL08
        """
        Sample the Rt values using a random walk process
        and broadcast them to daily values.

        Parameters
        ----------
        n_steps : int
            Number of time steps to sample.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample calls.

        Returns
        -------
        ArrayLike
            An array containing the broadcasted Rt values.
        """
        # sample the standard deviation for the random walk process
        sd_wt = numpyro.sample(
            "Wt_rw_sd", dist.HalfNormal(self.gamma_RW_prior_scale)
        )
        # Rt random walk process
        wt_rv = SimpleRandomWalkProcess(
            name="Wt",
            step_rv=DistributionalRV(
                name="rw_step_rv",
                dist=dist.Normal(0, sd_wt),
                reparam=LocScaleReparam(0),
            ),
            init_rv=DistributionalRV(
                name="init_Wt_rv",
                dist=self.intercept_RW_prior,
            ),
        )
        # transform Rt random walk w/ scaled logit
        transformed_rt_samples = TransformedRandomVariable(
            name="transformed_rt_rw",
            base_rv=wt_rv,
            transforms=t.ScaledLogitTransform(x_max=self.max_rt).inv,
        ).sample(n_steps=n_steps, **kwargs)
        # broadcast the Rt samples to daily values
        broadcasted_rt_samples = transformed_rt_samples[0].value[
            self.week_indices
        ]
        logging.debug(f"Broadcasted Rt samples: {broadcasted_rt_samples}")
        return broadcasted_rt_samples


class CFAEPIM_Model_Sample(NamedTuple):  # numpydoc ignore=GL08
    Rts: SampledValue | None = None
    latent_infections: SampledValue | None = None
    susceptibles: SampledValue | None = None
    ascertainment_rates: SampledValue | None = None
    expected_hospitalizations: SampledValue | None = None

    def __repr__(self):
        return (
            f"CFAEPIM_Model_Sample(Rts={self.Rts}, "
            f"latent_infections={self.latent_infections}, "
            f"susceptibles={self.susceptibles}, "
            f"ascertainment_rates={self.ascertainment_rates}, "
            f"expected_hospitalizations={self.expected_hospitalizations}"
        )


class CFAEPIM_Model(Model):
    """
    CFAEPIM Model class for epidemic inference,
    ported over from `cfaepim`. This class handles the
    initialization and sampling of the CFAEPIM model,
    including the transmission process, infection process,
    and observation process.

    Parameters
    ----------
    config : dict[str, any]
        Configuration dictionary containing model parameters.
    population : int
        Total population size.
    week_indices : ArrayLike
        Array of week indices corresponding to the time steps.
    first_week_hosp : int
        Number of hospitalizations in the first week.
    predictors : list[int]
        List of predictors (covariates) for the model.
    data_observed_hosp_admissions : pl.DataFrame
        DataFrame containing observed hospital admissions data.
    """

    def __init__(
        self,
        config: dict[str, any],
        population: int,
        week_indices: ArrayLike,
        first_week_hosp: int,
        predictors: list[int],
    ):  # numpydoc ignore=GL08
        self.population = population
        self.week_indices = week_indices
        self.first_week_hosp = first_week_hosp
        self.predictors = predictors

        self.config = config
        for key, value in config.items():
            setattr(self, key, value)

        # transmission: generation time distribution
        self.pmf_array = jnp.array(self.generation_time_dist)
        self.gen_int = DeterministicPMF(name="gen_int", value=self.pmf_array)
        # update: record in sample ought to be False by default

        # transmission: prior for RW intercept
        self.intercept_RW_prior = dist.Normal(
            self.rt_intercept_prior_mode, self.rt_intercept_prior_scale
        )

        # transmission: Rt process
        self.Rt_process = CFAEPIM_Rt(
            intercept_RW_prior=self.intercept_RW_prior,
            max_rt=self.max_rt,
            gamma_RW_prior_scale=self.weekly_rw_prior_scale,
            week_indices=self.week_indices,
        )

        # infections: get value rate for infection seeding (initialization)
        self.mean_inf_val = (
            self.inf_model_prior_infections_per_capita * self.population
        ) + (self.first_week_hosp / (self.ihr_intercept_prior_mode * 7))

        # infections: initial infections
        self.I0 = InfectionInitializationProcess(
            name="I0_initialization",
            I_pre_init_rv=DistributionalRV(
                name="I0",
                dist=dist.Exponential(rate=1 / self.mean_inf_val).expand(
                    [self.inf_model_seed_days]
                ),
            ),
            infection_init_method=InitializeInfectionsFromVec(
                n_timepoints=self.inf_model_seed_days
            ),
            t_unit=1,
        )

        # infections: susceptibility depletion prior
        # update: truncated Normal needed here, done
        # "under the hood" in Epidemia, use Beta for the
        # time being.
        # self.susceptibility_prior = dist.Beta(
        #     1
        #     + (
        #         self.susceptible_fraction_prior_mode
        #         / self.susceptible_fraction_prior_scale
        #     ),
        #     1
        #     + (1 - self.susceptible_fraction_prior_mode)
        #     / self.susceptible_fraction_prior_scale,
        # )
        # now:
        self.susceptibility_prior = dist.TruncatedNormal(
            self.susceptible_fraction_prior_mode,
            self.susceptible_fraction_prior_scale,
            low=0.0,
        )

        # infections component
        self.infections = CFAEPIM_Infections(
            I0=self.I0, susceptibility_prior=self.susceptibility_prior
        )

        # observations: negative binomial concentration prior
        self.nb_concentration_prior = dist.Normal(
            self.reciprocal_dispersion_prior_mode,
            self.reciprocal_dispersion_prior_scale,
        )

        # observations: instantaneous ascertainment rate prior
        self.alpha_prior_dist = dist.Normal(
            self.ihr_intercept_prior_mode, self.ihr_intercept_prior_scale
        )

        # observations: prior on covariate coefficients
        self.coefficient_priors = dist.Normal(
            loc=jnp.array(
                self.day_of_week_effect_prior_modes
                + [
                    self.holiday_eff_prior_mode,
                    self.post_holiday_eff_prior_mode,
                    self.non_obs_effect_prior_mode,
                ]
            ),
            scale=jnp.array(
                self.day_of_week_effect_prior_scales
                + [
                    self.holiday_eff_prior_scale,
                    self.post_holiday_eff_prior_scale,
                    self.non_obs_effect_prior_scale,
                ]
            ),
        )

        # observations component
        self.obs_process = CFAEPIM_Observation(
            predictors=self.predictors,
            alpha_prior_dist=self.alpha_prior_dist,
            coefficient_priors=self.coefficient_priors,
            nb_concentration_prior=self.nb_concentration_prior,
        )

    @staticmethod
    def validate(
        population: any,
        week_indices: any,
        first_week_hosp: any,
        predictors: any,
    ) -> None:
        """
        Validate the parameters of the CFAEPIM model.

        This method checks that all necessary parameters and priors are correctly specified.
        If any parameter is invalid, an appropriate error is raised.

        Raises
        ------
        ValueError
            If any parameter is missing or invalid.
        """
        if not isinstance(population, int) or population <= 0:
            raise ValueError("Population must be a positive integer.")
        if not isinstance(week_indices, jax.ndarray):
            raise ValueError("Week indices must be an array-like structure.")
        if not isinstance(first_week_hosp, int) or first_week_hosp < 0:
            raise ValueError(
                "First week hospitalizations must be a non-negative integer."
            )
        if not isinstance(predictors, jnp.ndarray):
            raise ValueError("Predictors must be a list of integers.")

    def sample(
        self,
        n_steps: int,
        data_observed_hosp_admissions: ArrayLike = None,
        **kwargs,
    ) -> tuple:
        # shift towards "reduced statefulness", include here week indices &
        # predictors which might change; for the same model and different
        # models.
        """
        Samples the reproduction numbers, generation interval,
        infections, and hospitalizations from the CFAEPIM model.

        Parameters
        ----------
        n_steps : int
            Number of time steps to sample.
        data_observed_hosp_admissions : ArrayLike, optional
            Observation hospital admissions.
            Defaults to None.
        **kwargs : dict, optional
            Additional keyword arguments passed through to
            internal sample calls, should there be any.

        Returns
        -------
        CFAEPIM_Model_Sample
            A named tuple containing sampled values for reproduction numbers,
            latent infections, susceptibles, ascertainment rates, expected
            hospitalizations, and observed hospital admissions.
        """
        sampled_Rts = self.Rt_process.sample(n_steps=n_steps)
        sampled_gen_int = self.gen_int.sample(record=False)
        all_I_t, all_S_t = self.infections.sample(
            Rt=sampled_Rts,
            gen_int=sampled_gen_int[0].value,
            P=self.population,
        )
        sampled_alphas, expected_hosps = self.obs_process.sample(
            infections=all_I_t,
            inf_to_hosp_dist=jnp.array(self.inf_to_hosp_dist),
        )
        # observed_hosp_admissions = self.obs_process.nb_observation.sample(
        #     mu=expected_hosps,
        #     obs=data_observed_hosp_admissions,
        #     **kwargs,
        # )
        numpyro.deterministic("Rts", sampled_Rts)
        numpyro.deterministic("latent_infections", all_I_t)
        numpyro.deterministic("susceptibles", all_S_t)
        numpyro.deterministic("alphas", sampled_alphas)
        numpyro.deterministic("expected_hospitalizations", expected_hosps)
        return CFAEPIM_Model_Sample(
            Rts=sampled_Rts,
            latent_infections=all_I_t,
            susceptibles=all_S_t,
            ascertainment_rates=sampled_alphas,
            expected_hospitalizations=expected_hosps,
        )


def add_post_observation_period(
    dataset: pl.DataFrame, n_post_observation_days: int
) -> pl.DataFrame:  # numpydoc ignore=RT01
    """
    Receives a dataframe that is filtered down to a
    particular jurisdiction, that has pre-observation
    data, and adds new rows to the end of the dataframe
    for the post-observation (forecasting) period.
    """

    # calculate the dates from the latest date in the dataframe
    max_date = dataset["date"].max()
    post_observation_dates = [
        (max_date + timedelta(days=i))
        for i in range(1, n_post_observation_days + 1)
    ]

    # get the days of the week (e.g. Fri) from the calculated dates
    day_of_weeks = (
        pl.Series(post_observation_dates)
        .dt.strftime("%a")
        .alias("day_of_week")
    )
    weekends = day_of_weeks.is_in(["Sat", "Sun"])

    # calculate the epiweeks and epiyears, which might not evenly mod 7
    last_epiweek = dataset["epiweek"][-1]
    epiweek_counts = dataset.filter(pl.col("epiweek") == last_epiweek).shape[0]
    epiweeks = [last_epiweek] * (7 - epiweek_counts) + [
        (last_epiweek + 1 + (i // 7))
        for i in range(n_post_observation_days - (7 - epiweek_counts))
    ]
    last_epiyear = dataset["epiyear"][-1]
    epiyears = [
        last_epiyear if epiweek <= 52 else last_epiyear + 1
        for epiweek in epiweeks
    ]
    epiweeks = [
        epiweek if epiweek <= 52 else epiweek - 52 for epiweek in epiweeks
    ]

    # calculate week values
    last_week = dataset["week"][-1]
    week_counts = dataset.filter(pl.col("week") == last_week).shape[0]
    weeks = [last_week] * (7 - week_counts) + [
        (last_week + 1 + (i // 7))
        for i in range(n_post_observation_days - (7 - week_counts))
    ]
    weeks = [week if week <= 52 else week - 52 for week in weeks]

    # calculate holiday series
    holidays = [datetime.strptime(elt, "%Y-%m-%d") for elt in HOLIDAYS]
    holidays_values = [date in holidays for date in post_observation_dates]
    post_holidays = [holiday + timedelta(days=1) for holiday in holidays]
    post_holiday_values = [
        date in post_holidays for date in post_observation_dates
    ]

    # fill in post-observation data entries, zero hospitalizations
    post_observation_data = pl.DataFrame(
        {
            "location": [dataset["location"][0]] * n_post_observation_days,
            "date": post_observation_dates,
            "hosp": [-9999] * n_post_observation_days,  # possible
            "epiweek": epiweeks,
            "epiyear": epiyears,
            "day_of_week": day_of_weeks,
            "is_weekend": weekends,
            "is_holiday": holidays_values,
            "is_post_holiday": post_holiday_values,
            "recency": [0] * n_post_observation_days,
            "week": weeks,
            "location_code": [dataset["location_code"][0]]
            * n_post_observation_days,
            "population": [dataset["population"][0]] * n_post_observation_days,
            "first_week_hosp": [dataset["first_week_hosp"][0]]
            * n_post_observation_days,
            "nonobservation_period": [False] * n_post_observation_days,
        }
    )

    # stack post_observation_data ONTO dataset
    merged_data = dataset.vstack(post_observation_data)
    return merged_data


def add_pre_observation_period(
    dataset: pl.DataFrame, n_pre_observation_days: int
) -> pl.DataFrame:  # numpydoc ignore=RT01
    """
    Receives a dataframe that is filtered down to a
    particular jurisdiction and adds new rows to the
    beginning of the dataframe for the non-observation
    period.
    """

    # create new nonobs column, set to False by default
    dataset = dataset.with_columns(
        pl.lit(False).alias("nonobservation_period")
    )

    # backcalculate the dates from the earliest date in the dataframe
    min_date = dataset["date"].min()
    pre_observation_dates = [
        (min_date - timedelta(days=i))
        for i in range(1, n_pre_observation_days + 1)
    ]
    pre_observation_dates.reverse()

    # get the days of the week (e.g. Fri) from the backcalculated dates
    day_of_weeks = (
        pl.Series(pre_observation_dates).dt.strftime("%a").alias("day_of_week")
    )
    weekends = day_of_weeks.is_in(["Sat", "Sun"])

    # backculate the epiweeks, which might not evenly mod 7
    first_epiweek = dataset["epiweek"][0]
    counts = dataset.filter(pl.col("epiweek") == first_epiweek).shape[0]
    epiweeks = [first_epiweek] * (7 - counts) + [
        (first_epiweek - 1 - (i // 7))
        for i in range(n_pre_observation_days - (7 - counts))
    ]
    epiweeks.reverse()

    # calculate holiday series
    holidays = [datetime.strptime(elt, "%Y-%m-%d") for elt in HOLIDAYS]
    holidays_values = [date in holidays for date in pre_observation_dates]
    post_holidays = [holiday + timedelta(days=1) for holiday in holidays]
    post_holiday_values = [
        date in post_holidays for date in pre_observation_dates
    ]

    # fill in pre-observation data entries, zero hospitalizations
    pre_observation_data = pl.DataFrame(
        {
            "location": [dataset["location"][0]] * n_pre_observation_days,
            "date": pre_observation_dates,
            "hosp": [0] * n_pre_observation_days,
            "epiweek": epiweeks,
            "epiyear": [dataset["epiyear"][0]] * n_pre_observation_days,
            "day_of_week": day_of_weeks,
            "is_weekend": weekends,
            "is_holiday": holidays_values,
            "is_post_holiday": post_holiday_values,
            "recency": [0] * n_pre_observation_days,
            "week": [dataset["week"][0]] * n_pre_observation_days,
            "location_code": [dataset["location_code"][0]]
            * n_pre_observation_days,
            "population": [dataset["population"][0]] * n_pre_observation_days,
            "first_week_hosp": [dataset["first_week_hosp"][0]]
            * n_pre_observation_days,
            "nonobservation_period": [True] * n_pre_observation_days,
        }
    )

    # stack dataset ONTO pre_observation_data
    merged_data = pre_observation_data.vstack(dataset)
    return merged_data


def process_jurisdictions(value):  # numpydoc ignore=GL08
    if value.lower() == "all":
        return JURISDICTIONS
    elif value.lower().startswith("not:"):
        exclude = value[4:].split(",")
        return [state for state in JURISDICTIONS if state not in exclude]
    else:
        return value.split(",")


# def instantiate_CFAEPIM(
#     jurisdiction: str,
#     dataset: pl.DataFrame,
#     config: dict[str, any],
#     forecasting: bool = False,
#     n_post_observation_days: int = 0,
# ):  # numpydoc ignore=GL08
#     """
#     Originally separated to support `model_render`;
#     possibly reintegrated w/ `run_single_jurisdiction`.
#     """


def run_single_jurisdiction(
    jurisdiction: str,
    dataset: pl.DataFrame,
    config: dict[str, any],
    forecasting: bool = False,
    n_post_observation_days: int = 0,
):
    """
    Runs the ported `cfaepim` model on a single
    jurisdiction. Pre- and post-observation data
    for the Rt burn in and for forecasting,
    respectively, is done before the prior predictive,
    posterior, and posterior predictive samples
    are returned.

    Parameters
    ----------
    jurisdiction : str
        The jurisdiction.
    dataset : pl.DataFrame
        The incidence data of interest.
    config : dict[str, any]
        A configuration file for the model.
    forecasting : bool, optional
        Whether or not forecasts are being made.
        Defaults to True.
    n_post_observation_days : int, optional
        The number of days to look ahead. Defaults
        to 0 if not forecasting.

    Returns
    -------
    tuple
        A tuple of prior predictive, posterior, and
        posterior predictive samples.
    """
    # filter data to be the jurisdiction alone
    filtered_data_jurisdiction = dataset.filter(
        pl.col("location") == jurisdiction
    )

    # add the pre-observation period to the dataset
    filtered_data = add_pre_observation_period(
        dataset=filtered_data_jurisdiction,
        n_pre_observation_days=config["n_pre_observation_days"],
    )

    logging.info(f"{jurisdiction}: Dataset w/ pre-observation ready.")

    if forecasting:
        # add the post-observation period if forecasting
        filtered_data = add_post_observation_period(
            dataset=filtered_data,
            n_post_observation_days=n_post_observation_days,
        )
        logging.info(f"{jurisdiction}: Dataset w/ post-observation ready.")

    # extract jurisdiction population
    population = (
        filtered_data.select(pl.col("population"))
        .unique()
        .to_numpy()
        .flatten()
    )[0]

    # extract indices for weeks for Rt broadcasting (weekly to daily)
    week_indices = filtered_data.select(pl.col("week")).to_numpy().flatten()

    # extract first week hospitalizations for infections seeding
    first_week_hosp = (
        filtered_data.select(pl.col("first_week_hosp"))
        .unique()
        .to_numpy()
        .flatten()
    )[0]

    # extract covariates (typically weekday, holidays, nonobs period)
    day_of_week_covariate = (
        filtered_data.select(pl.col("day_of_week"))
        .to_dummies()
        .select(pl.exclude("day_of_week_Thu"))
    )
    remaining_covariates = filtered_data.select(
        ["is_holiday", "is_post_holiday", "nonobservation_period"]
    )
    covariates = pl.concat(
        [day_of_week_covariate, remaining_covariates], how="horizontal"
    )
    predictors = covariates.to_numpy()

    # extract observation hospital admissions
    # NOTE: from filtered_data_jurisdiction, not filtered_data, which has null hosp
    observed_hosp_admissions = (
        filtered_data.select(pl.col("hosp")).to_numpy().flatten()
    )

    logging.info(f"{jurisdiction}: Variables extracted from dataset.")

    # instantiate CFAEPIM model (for fitting)
    total_steps = week_indices.size
    steps_excluding_forecast = total_steps - n_post_observation_days
    cfaepim_MSR_fit = CFAEPIM_Model(
        config=config,
        population=population,
        week_indices=week_indices[:steps_excluding_forecast],
        first_week_hosp=first_week_hosp,
        predictors=predictors[:steps_excluding_forecast],
    )

    logging.info(f"{jurisdiction}: CFAEPIM model instantiated (fitting)!")

    # run the CFAEPIM model
    cfaepim_MSR_fit.run(
        n_steps=steps_excluding_forecast,
        data_observed_hosp_admissions=observed_hosp_admissions[
            :steps_excluding_forecast
        ],
        num_warmup=config["n_warmup"],
        num_samples=config["n_iter"],
        nuts_args={
            "target_accept_prob": config["adapt_delta"],
            "max_tree_depth": config["max_treedepth"],
        },
        mcmc_args={
            "num_chains": config["n_chains"],
            "progress_bar": True,
        },  # progress_bar False if use vmap
    )

    logging.info(f"{jurisdiction}: CFAEPIM model (fitting) ran!")

    cfaepim_MSR_fit.print_summary()

    # prior predictive simulation samples
    prior_predictive_sim_samples = cfaepim_MSR_fit.prior_predictive(
        n_steps=steps_excluding_forecast,
        numpyro_predictive_args={"num_samples": config["n_iter"]},
        rng_key=jax.random.key(config["seed"]),
    )

    logging.info(f"{jurisdiction}: Prior predictive simulation complete.")

    # posterior predictive simulation samples
    posterior_predictive_sim_samples = cfaepim_MSR_fit.posterior_predictive(
        n_steps=steps_excluding_forecast,
        numpyro_predictive_args={"num_samples": config["n_iter"]},
        rng_key=jax.random.key(config["seed"]),
        data_observed_hosp_admissions=None,
    )

    logging.info(f"{jurisdiction}: Posterior predictive simulation complete.")

    # posterior predictive forecasting samples
    if forecasting:
        cfaepim_MSR_for = CFAEPIM_Model(
            config=config,
            population=population,
            week_indices=week_indices,
            first_week_hosp=first_week_hosp,
            predictors=predictors,
        )

        # run the CFAEPIM model (forecasting, required to do so
        # single `posterior_predictive` gets sames (need self.mcmc)
        # from passed model);
        # ISSUE: inv()
        # PR: sample() + OOP behavior & statefulness
        cfaepim_MSR_for.mcmc = cfaepim_MSR_fit.mcmc

        posterior_predictive_for_samples = (
            cfaepim_MSR_for.posterior_predictive(
                n_steps=total_steps,
                numpyro_predictive_args={"num_samples": config["n_iter"]},
                rng_key=jax.random.key(config["seed"]),
                data_observed_hosp_admissions=None,
            )
        )

        logging.info(
            f"{jurisdiction}: Posterior predictive forecasts complete."
        )
    else:
        posterior_predictive_for_samples = None

    return (
        cfaepim_MSR_for,
        observed_hosp_admissions,
        prior_predictive_sim_samples,
        posterior_predictive_sim_samples,
        posterior_predictive_for_samples,
    )


def save_numpyro_model(
    save_path: str,
    jurisdiction: str,
    dataset: pl.DataFrame,
    config: dict[str, any],
    forecasting: bool = False,
    n_post_observation_days: int = 0,
):  # numpydoc ignore=GL08
    # check if the file exists
    if os.path.exists(save_path):
        pass

    else:
        # instantiate cfaepim_MSR
        # _, _, cfaepim_MSR, _ = instantiate_CFAEPIM(
        #     jurisdiction, dataset, config, forecasting, n_post_observation_days
        # )
        cfaepim_MSR = ""
        # needs to be fixed; sample
        # then pass model + args to model

        render_model(cfaepim_MSR, filename=save_path)


def template_save_file(
    title: str, save_path: str, figure_and_descriptions: list[tuple[str, str]]
):  # numpydoc ignore=GL08
    header_p1 = f"""
    ---
    title: "{title}"
    author: "CFA"
    date: "{CURRENT_DATE}"
    """
    header_p2 = open("header_p2.txt").read()
    content = header_p1 + header_p2 + "\n"
    for plot_title, plot_path in figure_and_descriptions:
        content += f"""
        ![{plot_title}]({plot_path}){{ width=75% }}\n
        """
    if not os.path.exists(save_path):
        with open(save_path, "w") as f:
            f.write(content)
            f.close()


def convert_markdown_output_files_to_pdf():  # numpydoc ignore=GL08
    markdown_files = glob.glob("*.md")
    print(markdown_files)
    pass


def save_inference_content():
    """
    Saves MCMC inference content.
    """
    pass


def plot_lm_arviz_fit(idata):  # numpydoc ignore=GL08
    fig, ax = plt.subplots()
    az.plot_lm(
        "negbinom_rv",
        idata=idata,
        kind_pp="hdi",
        y_kwargs={"color": "black"},
        y_hat_fill_kwargs={"color": "C0"},
        axes=ax,
    )
    ax.set_title("Posterior Predictive Plot")
    ax.set_ylabel("Hospital Admissions")
    ax.set_xlabel("Days")
    plt.show()


def compute_eti(dataset, eti_prob):  # numpydoc ignore=GL08
    eti_bdry = dataset.quantile(
        ((1 - eti_prob) / 2, 1 / 2 + eti_prob / 2), dim=("chain", "draw")
    )
    return eti_bdry.values.T


def plot_hdi_arviz_for(idata, forecast_days):  # numpydoc ignore=GL08
    x_data = idata.posterior_predictive["negbinom_rv_dim_0"] + forecast_days
    y_data = idata.posterior_predictive["negbinom_rv"]
    fig, axes = plt.subplots(figsize=(6, 5))
    az.plot_hdi(
        x_data,
        hdi_data=compute_eti(y_data, 0.9),
        color="C0",
        smooth=False,
        fill_kwargs={"alpha": 0.3},
        ax=axes,
    )

    az.plot_hdi(
        x_data,
        hdi_data=compute_eti(y_data, 0.5),
        color="C0",
        smooth=False,
        fill_kwargs={"alpha": 0.6},
        ax=axes,
    )
    median_ts = y_data.median(dim=["chain", "draw"])
    plt.plot(
        x_data,
        median_ts,
        color="C0",
        label="Median",
    )
    plt.scatter(
        idata.observed_data["negbinom_rv_dim_0"] + forecast_days,
        idata.observed_data["negbinom_rv"],
        color="black",
    )
    axes.legend()
    axes.set_title(
        "Posterior Predictive Admissions, including a forecast", fontsize=10
    )
    axes.set_xlabel("Time", fontsize=10)
    axes.set_ylabel("Hospital Admissions", fontsize=10)
    plt.show()


# def quantilize_forecasts(
#     samples_dict,
#     state_abbr,
#     start_date,
#     end_date,
#     fitting_data,
#     output_path,
#     reference_date,
# ):  # numpydoc ignore=GL08
#     pandas2ri.activate()
#     forecasttools = importr("forecasttools")
#     # dplyr = importr("dplyr")
#     # tidyr = importr("tidyr")
#     # cli = importr("cli")

#     posterior_samples = pl.DataFrame(samples_dict)
#     posterior_samples_pd = posterior_samples.to_pandas()
#     r_posterior_samples = pandas2ri.py2rpy(posterior_samples_pd)

#     fitting_data_pd = fitting_data.to_pandas()
#     r_fitting_data = pandas2ri.py2rpy(fitting_data_pd)

#     results_list = ro.ListVector({state_abbr: r_posterior_samples})

#     horizons = ro.IntVector([-1, 0, 1, 2, 3])

#     forecast_output = forecasttools.forecast_and_output_flusight(
#         data=r_fitting_data,
#         results=results_list,
#         output_path=output_path,
#         reference_date=reference_date,
#         horizons=horizons,
#         seed=62352,
#     )

#     forecast_output_pd = pandas2ri.rpy2py(forecast_output)
#     forecast_output_pl = pl.from_pandas(forecast_output_pd)
#     print(forecast_output_pl)


def main(args):  # numpydoc ignore=GL08
    """
    The `cfaepim` model required a configuration
    file and a dataset. The configuration file must
    follow some detailed specifications, as must the
    dataset. Once these are in place, the model is
    used in the following manner for each state:
    (1) extract the population, the indices of the weeks,
    the hospitalizations during the first week, & the
    covariates, (2) the configuration file and the
    previous content then will be used to produce
    an Rt, infections, and observation process by
    passing them to the `cfaepim` model, (3) the user
    can use argparse to test or compare the forecasts.
    The `cfaepim` tool is used for runs on hospitalization
    data retrieved from an API or stored historically.

    Notes
    -----
    Testing in `cfaepim` includes ensuring the dataset
    and configuration have the correct variables and
    values in a proper range. Testing also ensures that
    each part of the `cfaepim` model works as desired.
    python3 tut_epim_port_msr.py --reporting_date 2024-01-20 --regions NY --historical --forecast
    """
    logging.info("Starting CFAEPIM")

    # determine number of CPU cores
    numpyro.set_platform("cpu")
    num_cores = os.cpu_count()
    numpyro.set_host_device_count(num_cores - (num_cores - 3))
    logging.info("Number of cores set.")

    # check that output directory exists, if not create
    output_directory = ensure_output_directory(args)
    print(output_directory)
    logging.info("Output directory ensured working.")

    if args.historical_data:
        # check that historical cfaepim data exists for given reporting date
        historical_data_directory = assert_historical_data_files_exist(
            args.reporting_date
        )

        # load historical configuration file (modified from cfaepim)
        config = load_config(
            config_path=f"./config/params_{args.reporting_date}_historical.toml"
        )
        logging.info("Configuration (historical) loaded.")

        # load the historical hospitalization data
        data_path = os.path.join(
            historical_data_directory, f"{args.reporting_date}_clean_data.tsv"
        )
        influenza_hosp_data = load_data(data_path=data_path)
        logging.info("Incidence data (historical) loaded.")
        _, cols = influenza_hosp_data.shape
        display_data(
            data=influenza_hosp_data, n_row_count=10, n_col_count=cols
        )

        # modify date column from str to datetime
        influenza_hosp_data = influenza_hosp_data.with_columns(
            pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
        )

        # save plots of the raw hospitalization data,
        # for all jurisdictions
        if args.data_info_save:
            # save pdf of 2, 2x2 (log-scale plots)
            # total hospitalizations (full season) & last 4 weeks
            # log scale, log scale
            # growth rate, moving average
            # log-scale, log-scale
            # check if this already exist + do for all juris.
            pass

        if args.model_info_save:
            # save model diagram
            # save plots for priors
            # check if this already exists, do for each config file
            save_numpyro_model(
                save_path=output_directory + "cfaepim_diagram.pdf",
                jurisdiction="NY",
                dataset=influenza_hosp_data,
                config=config,
                forecasting=args.forecast,
                n_post_observation_days=28,
            )

        # parallel run over jurisdictions
        # results = dict([(elt, {}) for elt in args.regions])
        forecast_days = 28
        for jurisdiction in args.regions:
            # check if a folder for the samples exists
            # check if a folder for the jurisdiction exists

            # assumptions, fit, and forecast for each jurisdiction
            (
                model,
                obs,
                prior_p_ss,
                post_p_ss,
                post_p_fs,
            ) = run_single_jurisdiction(
                jurisdiction=jurisdiction,
                dataset=influenza_hosp_data,
                config=config,
                forecasting=args.forecast,
                n_post_observation_days=forecast_days,
            )

            idata = az.from_numpyro(
                posterior=model.mcmc,
                prior=prior_p_ss,
                posterior_predictive=post_p_ss,
                constant_data={"obs": obs},
            )
            print(dir(idata))
            # plot_lm_arviz_fit(idata)
            plot_hdi_arviz_for(idata, forecast_days)

            # save to folder for jurisdiction,

            # idata = az.from_numpyro(model.mcmc)
            # diagnostic_stats_summary = az.summary(
            #     idata.posterior,
            #     kind="diagnostics",
            # )
            # print(diagnostic_stats_summary.loc["negbinom_rv"])

            # ax.set_title("Posterior Predictive Plot")
            # ax.set_ylabel("Hospital Admissions")
            # ax.set_xlabel("Days")
            # plt.show()

            # prior_p_ss_figures_and_descriptions = plot_sample_variables(
            #     samples=prior_p_ss,
            #     variables=["Rts", "latent_infections", "negbinom_rv"],
            #     observations=obs,
            #     ylabels=[
            #         "Basic Reproduction Number",
            #         "Latent Infections",
            #         "Hospital Admissions",
            #     ],
            #     plot_types=["TRACE", "PPC", "HDI"],
            #     plot_kwargs={
            #         "HDI": {"hdi_prob": 0.95, "plot_kwargs": {"ls": "-."}},
            #         "TRACE": {"var_names": ["Rts", "latent_infections"]},
            #         "PPC": {"alpha": 0.05, "textsize": 12},
            #     },
            # )

            # print(prior_p_ss_figures_and_descriptions)

        # if args.forecasting:

    # prior_p_ss & post_p_ss get their own pdf (markdown first then subprocess)
    # each variable is plotted out, if possible
    # arviz diagnostics


if __name__ == "__main__":
    # argparse settings
    # e.g. python3 tut_epim_port_msr.py
    # --reporting_date 2024-01-20 --regions all --historical --forecast
    parser = argparse.ArgumentParser(
        description="Forecast, simulate, and analyze the CFAEPIM model."
    )
    parser.add_argument(
        "--regions",
        type=process_jurisdictions,
        required=True,
        help="Specify jurisdictions as a comma-separated list. Use 'all' for all states, or 'not:state1,state2' to exclude specific states.",
    )
    parser.add_argument(
        "--reporting_date",
        type=str,
        required=True,
        help="The reporting date.",
    )
    parser.add_argument(
        "--historical_data",
        action="store_true",
        help="Load model weights before training.",
    )
    parser.add_argument(
        "--forecast",
        action="store_true",
        help="Whether to make a forecast.",
    )
    parser.add_argument(
        "--data_info_save",
        action="store_true",
        help="Whether to save information about the dataset.",
    )
    parser.add_argument(
        "--model_info_save",
        action="store_true",
        help="Whether to save information about the model.",
    )
    args = parser.parse_args()
    main(args)

# TODO
# argparse
#   turn off reports
# report(s) generation
# plotting generation
# generalized plotting
# forecasttools formatting
# MCMC utils (numpyro)
# issues x 3 (plotting, inv(), infections docs.)
# tests
# Save MCMC + Samples
# Save to Image as Metadata
# forecast scoring & interpretation
#   in reports
#   probabilistic statements on what to expect
#   relative to historical (several weeks prior + last year)
# evaluate configuration file
# tutorial on usage
# writing again
# notes about what each function must know
# plot objects include: latent infections, observed hospital
# admissions, Rt; plot types includes: dist., density, posterior,
# density comparison, pair plot, posterior predictive check plot,
# HDI plot,
