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
from datetime import datetime
from typing import NamedTuple

# import arviz as az
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


# set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Config:  # numpydoc ignore=GL08
    def __init__(self, config_dict):  # numpydoc ignore=GL08
        for key, value in config_dict.items():
            setattr(self, key, value)


def display_data(
    data: pl.DataFrame, n_row_count: int = 15, n_col_count: int = 5
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
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_hide_column_data_types(True)
    with pl.Config(tbl_rows=n_row_count, tbl_cols=n_col_count):
        print(f"Dataset In Use For `cfaepim`:\n{data}\n")


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


def forecast_report():  # numpydoc ignore=GL08
    pass


def model_report():  # numpydoc ignore=GL08
    pass


def prior_report():  # numpydoc ignore=GL08
    pass


def plot_utils(
    axes: mpl.axes.Axes,
    figure: plt.Figure,
    use_log: bool = False,
    title: str = "",
    ylabel: str = "",
    xlabel: str = "",
    use_legend: bool = False,
    display: bool = True,
    filename: str = "delete_me",
    save_as_img: bool = False,
    save_to_pdf: bool = False,
) -> None | plt.Figure:  # numpydoc ignore=GL08
    """
    Utility function to format and save plots.

    Parameters
    ----------
    axes : mpl.axes.Axes
        The axes object to format.
    figure : plt.Figure
        The figure object to format and possibly save.
    use_log : bool, optional
        Whether to use log-scaling on the y-axis.
        Defaults to False.
    title : str, optional
        Title of the plot.
        Defaults to "".
    ylabel : str, optional
        Label for the y-axis.
        Defaults to "".
    xlabel : str, optional
        Label for the x-axis.
        Defaults to "".
    use_legend : bool, optional
        Whether to display a legend.
        Defaults to False.
    display : bool, optional
        Whether to display the plot.
        Defaults to True.
    filename : str, optional
        Filename for saving the plot.
        Defaults to "delete_me".
    save_as_img : bool, optional
        Whether to save the plot as an image.
        Defaults to False.
    save_to_pdf : bool, optional
        Whether to return the figure for saving as a PDF.
        Defaults to False.

    Returns
    -------
    None | plt.Figure
        Returns the figure if save_to_pdf is True, otherwise returns None.
    """
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
) -> None | plt.Figure:  # numpydoc ignore=GL08
    """
    Basic plot function for y vs X.

    Parameters
    ----------
    y : np.ndarray
        Data for the y-axis.
    X : np.ndarray
        Data for the x-axis.
    title : str, optional
        Title of the plot. Defaults to "".
    X_label : str, optional
        Label for the x-axis. Defaults to "".
    Y_label : str, optional
        Label for the y-axis. Defaults to "".
    use_log : bool, optional
        Whether to use log-scaling on the y-axis. Defaults to False.
    use_legend : bool, optional
        Whether to display a legend. Defaults to False.
    display : bool, optional
        Whether to display the plot. Defaults to True.
    filename : str, optional
        Filename for saving the plot. Defaults to "delete_me".
    save_as_img : bool, optional
        Whether to save the plot as an image. Defaults to False.
    save_to_pdf : bool, optional
        Whether to return the figure for saving as a PDF. Defaults to False.

    Returns
    -------
    None | plt.Figure
        Returns the figure if save_to_pdf is True, otherwise returns None.
    """
    figure, axes = plt.subplots(1, 1)
    axes.plot(X, y, color="black")
    axes.set_xlim(left=0)
    axes.set_ylim(bottom=0)
    return plot_utils(
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


def plot_prior_distributions(
    prior_distributions: dict,
    num_samples: int = 1000,
    save_as_img: bool = False,
    save_to_pdf: bool = False,
    display: bool = True,
) -> None:
    """
    Plot prior distributions and save them as images or PDF.

    Parameters
    ----------
    prior_distributions : dict
        Dictionary of prior distributions to plot.
    num_samples : int, optional
        Number of samples to draw from each distribution. Defaults to 1000.
    save_as_img : bool, optional
        Whether to save the plots as images. Defaults to False.
    save_to_pdf : bool, optional
        Whether to save the plots to a PDF. Defaults to False.
    display : bool, optional
        Whether to display the plots. Defaults to True.

    Returns
    -------
    None
    """
    figures = []
    for name, distribution in prior_distributions.items():
        samples = distribution.sample(jax.random.PRNGKey(0), (num_samples,))
        figure, axes = plt.subplots(1, 1)
        axes.hist(samples, bins=50, density=True, alpha=0.6, color="g")
        title = f"Prior Distribution: {name}"
        ylabel = "Density"
        xlabel = "Value"
        fig = plot_utils(
            axes=axes,
            figure=figure,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            use_legend=False,
            display=display,
            filename=name,
            save_as_img=save_as_img,
            save_to_pdf=save_to_pdf,
        )
        if save_to_pdf:
            figures.append(fig)
    # change further
    # if save_to_pdf:
    #     from matplotlib.backends.backend_pdf import PdfPages
    #     with PdfPages('./figures/prior_distributions.pdf') as pdf:
    #         for fig in figures:
    #             pdf.savefig(fig)
    return None


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

        # ensure the proportion is between 0 and 1
        init_S_proportion = jnp.clip(init_S_proportion, 0, 1)

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
        (_, all_S_t), all_I_t = numpyro.contrib.control_flow.scan(
            update_infections, init_carry, Rt
        )

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
        return broadcasted_rt_samples


class CFAEPIM_Model_Sample(NamedTuple):  # numpydoc ignore=GL08
    Rts: SampledValue | None = None
    latent_infections: SampledValue | None = None
    susceptibles: SampledValue | None = None
    ascertainment_rates: SampledValue | None = None
    expected_hospitalizations: SampledValue | None = None
    observed_hospital_admissions: SampledValue | None = None

    def __repr__(self):
        return (
            f"CFAEPIM_Model_Sample(Rts={self.Rts}, "
            f"latent_infections={self.latent_infections}, "
            f"susceptibles={self.susceptibles}, "
            f"ascertainment_rates={self.ascertainment_rates}, "
            f"expected_hospitalizations={self.expected_hospitalizations} ",
            f"observed_hospital_admissions={self.observed_hospital_admissions}",
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
        data_observed_hosp_admissions: pl.DataFrame,
    ):  # numpydoc ignore=GL08
        self.population = population
        self.week_indices = week_indices
        self.first_week_hosp = first_week_hosp
        self.predictors = predictors
        self.data_observed_hosp_admissions = data_observed_hosp_admissions

        self.config = config
        for key, value in config.items():
            setattr(self, key, value)

        # transmission: generation time distribution
        self.pmf_array = jnp.array(self.generation_time_dist)
        self.gen_int = DeterministicPMF(name="gen_int", value=self.pmf_array)

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
        self.susceptibility_prior = dist.Normal(
            self.susceptible_fraction_prior_mode,
            self.susceptible_fraction_prior_scale,
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
        data_observed_hosp_admissions: any,
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
        if not isinstance(data_observed_hosp_admissions, jnp.ndarray):
            raise ValueError(
                "Observed hospital admissions must be a jax array."
            )

        CFAEPIM_Model.infections.validate()
        CFAEPIM_Model.obs_process.validate()
        CFAEPIM_Model.Rt_process.validate()

    def sample(
        self,
        n_steps: int,
        **kwargs,
    ) -> tuple:
        """
        Samples the reproduction numbers, generation interval,
        infections, and hospitalizations from the CFAEPIM model.

        Parameters
        ----------
        n_steps : int
            Number of time steps to sample.
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
        sampled_gen_int = self.gen_int.sample()
        all_I_t, all_S_t = self.infections.sample(
            Rt=sampled_Rts,
            gen_int=sampled_gen_int[0].value,
            P=self.population,
        )
        sampled_alphas, expected_hosps = self.obs_process.sample(
            infections=all_I_t,
            inf_to_hosp_dist=jnp.array(self.inf_to_hosp_dist),
        )
        observed_hosp_admissions = self.obs_process.nb_observation.sample(
            mu=expected_hosps,
            obs=self.data_observed_hosp_admissions,
            **kwargs,
        )
        numpyro.deterministic("Rts", sampled_Rts)
        numpyro.deterministic("latent_infections", all_I_t)
        numpyro.deterministic("susceptibles", all_S_t)
        numpyro.deterministic("alphas", sampled_alphas)
        numpyro.deterministic("expected_hospitalizations", expected_hosps)
        numpyro.deterministic(
            "observed_hospitalizations", observed_hosp_admissions[0].value
        )
        return CFAEPIM_Model_Sample(
            Rts=sampled_Rts,
            latent_infections=all_I_t,
            susceptibles=all_S_t,
            ascertainment_rates=sampled_alphas,
            expected_hospitalizations=expected_hosps,
            observed_hospital_admissions=observed_hosp_admissions[0].value,
        )


def prepare_jurisdiction_data_jax(
    states: list[str], dataset: pl.DataFrame
):  # numpydoc ignore=GL08
    filtered_data = dataset.filter(pl.col("location").is_in(states))
    populations = (
        filtered_data.select(pl.col("population"))
        .unique()
        .to_numpy()
        .flatten()
    )
    week_indices = (
        filtered_data.select(pl.col("week"))
        .to_numpy()
        .reshape(len(states), -1)
    )
    first_week_hosps = (
        filtered_data.select(pl.col("first_week_hosp"))
        .unique()
        .to_numpy()
        .flatten()
    )
    day_of_week_covariate = filtered_data.select(
        pl.col("day_of_week")
    ).to_dummies()
    remaining_covariates = filtered_data.select(
        ["is_holiday", "is_post_holiday"]
    )
    covariates = pl.concat(
        [day_of_week_covariate, remaining_covariates], how="horizontal"
    )
    predictors = covariates.to_numpy().reshape(
        len(states), -1, covariates.shape[1]
    )
    data_observed_hosp_admissions = (
        filtered_data.select(pl.col("hosp"))
        .to_numpy()
        .reshape(len(states), -1)
    )
    return (
        jnp.array(populations),
        jnp.array(week_indices),
        jnp.array(first_week_hosps),
        jnp.array(predictors),
        jnp.array(data_observed_hosp_admissions),
    )


def run_batched(
    states: list[str], dataset: pl.DataFrame, config: dict[str, any]
):  # numpydoc ignore=GL08
    (
        populations,
        week_indices,
        first_week_hosps,
        predictors,
        data_observed_hosp_admissions,
    ) = prepare_jurisdiction_data_jax(states, dataset)

    def run_single(
        population,
        week_index,
        first_week_hosp,
        predictor,
        data_observed_hosp_admission,
    ):  # numpydoc ignore=GL08
        cfaepim_MSR = CFAEPIM_Model(
            config=config,
            population=population,
            week_indices=week_index,
            first_week_hosp=first_week_hosp,
            predictors=predictor,
            data_observed_hosp_admissions=data_observed_hosp_admission,
        )
        cfaepim_MSR.run(
            n_steps=week_indices.size,
            num_warmup=config["n_warmup"],
            num_samples=config["n_iter"],
            nuts_args={
                "target_accept_prob": config["adapt_delta"],
                "max_tree_depth": config["max_treedepth"],
            },
            mcmc_args={
                "num_chains": config["n_chains"],
                "progress_bar": False,
            },  # needs to be False to support vmap usage
        )
        cfaepim_MSR.print_summary()
        posterior_predictive_samples = cfaepim_MSR.posterior_predictive(
            n_steps=week_indices.size,
            numpyro_predictive_args={"num_samples": config["n_iter"]},
            rng_key=jax.random.key(config["seed"]),
        )
        prior_predictive_samples = cfaepim_MSR.prior_predictive(
            n_steps=week_indices.size,
            numpyro_predictive_args={"num_samples": config["n_iter"]},
            rng_key=jax.random.key(config["seed"]),
        )
        return (
            prior_predictive_samples,
            posterior_predictive_samples,
            cfaepim_MSR,
        )

    # run_single_vmap = jax.vmap(run_single, in_axes=(0, 0, 0, 0, 0))
    return jax.vmap(run_single)(
        populations,
        week_indices,
        first_week_hosps,
        predictors,
        data_observed_hosp_admissions,
    )
    # return jax.pmap(run_single)(jnp.arange(len(states)))


def main():  # numpydoc ignore=GL08
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
    """

    parser = argparse.ArgumentParser(
        description="Forecast, simulate, and analyze the CFAEPIM model."
    )
    parser.add_argument(
        "--reporting_date",
        type=str,
        required=True,
        help="The reporting date.",
    )
    # parser.add_argument(
    #     "--config_path",
    #     type=str,
    #     required=True,
    #     help="Crowd forecasts (sep. w/ space).",
    # )
    # parser.add_argument(
    #     "--data_path",
    #     type=str,
    #     required=True,
    #     help="Crowd forecasts (sep. w/ space).",
    # )
    parser.add_argument(
        "--historical_data",
        action="store_true",
        help="Load model weights before training.",
    )
    args = parser.parse_args()

    # determine number of CPU cores
    numpyro.set_platform("cpu")
    num_cores = os.cpu_count()
    numpyro.set_host_device_count(num_cores - (num_cores - 3))

    # data, output, and config directory
    # output_directory = ensure_output_directory(args)
    if args.historical_data:
        historical_data_directory = assert_historical_data_files_exist(
            args.reporting_date
        )
        config = load_config(
            config_path=f"./config/params_{args.reporting_date}_historical.toml"
        )
        data_path = os.path.join(
            historical_data_directory, f"{args.reporting_date}_clean_data.tsv"
        )
        influenza_hosp_data = load_data(data_path=data_path)
    if not args.historical_data:
        config = load_config(
            config_path=f"./config/params_{args.reporting_date}_historical.toml"
        )
    # influenza_hosp_data = influenza_hosp_data.with_column(
    #     pl.col("location_code").cast(pl.Int32)
    # )
    rows, cols = influenza_hosp_data.shape
    display_data(data=influenza_hosp_data, n_row_count=10, n_col_count=cols)

    # parallelized run over states
    states = ["NY", "CA"]
    results = run_batched(
        states=states, dataset=influenza_hosp_data, config=config
    )
    (
        prior_predictive_samples,
        posterior_predictive_samples,
        cfaepim_MSR,
    ) = results[0]
    print(prior_predictive_samples)
    # plot_results(
    #     prior_predictive_samples,
    #     posterior_predictive_samples,
    #     cfaepim_MSR)

    # plot data: for each state,

    # verification: plot single state hospitalizations
    # plot_single_location_hosp_data(
    #     incidence_data=influenza_hosp_data,
    #     states=["NY"],
    #     lower_date="2022-01-01",
    #     upper_date="2024-03-10",
    #     use_log=False,
    #     use_legend=True,
    #     save_as_img=False,
    #     save_to_pdf=False,
    #     display=False,
    # )

    # # verification: data_observed_hosp_admissions
    # print(f"HOSPITALIZATIONS:\n{data_observed_hosp_admissions}\n\n")

    # # verify and visualize aspects of the model
    # verify_cfaepim_MSR(cfaepim_MSR)


if __name__ == "__main__":
    main()
