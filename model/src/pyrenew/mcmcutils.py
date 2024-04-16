# -*- coding: utf-8 -*-

"""
Utilities to deal with MCMC outputs
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from jax.typing import ArrayLike


def spread_draws(
    posteriors: dict,
    variables_names: list,
) -> pl.DataFrame:
    """Get nicely shaped draws from the posterior

    Given a dictionary of posteriors, return a long-form polars dataframe
    indexed by draw, with variable values (equivalent of tidybayes
    spread_draws() function).

    Parameters
    ----------
    posteriors: dict
        A dictionary of posteriors with variable names as keys and numpy
        ndarrays as values (with the first axis corresponding to the posterior
        draw number.
    variables_names: list[str] | list[tuple]
        list of strings or of tuples identifying which variables to retrieve.

    Returns
    -------
    polars.DataFrame
    """

    for i_var, v in enumerate(variables_names):
        if isinstance(v, str):
            v_dims = None
        else:
            v_dims = v[1:]
            v = v[0]

        post = posteriors.get(v)
        long_post = post.flatten()[..., np.newaxis]

        indices = np.array(list(np.ndindex(post.shape)))
        n_dims = indices.shape[1] - 1
        if v_dims is None:
            dim_names = [
                ("{}_dim_{}_index".format(v, k), pl.Int64)
                for k in range(n_dims)
            ]
        elif len(v_dims) != n_dims:
            raise ValueError(
                "incorrect number of "
                "dimension names "
                "provided for variable "
                "{}".format(v)
            )
        else:
            dim_names = [(v_dim, pl.Int64) for v_dim in v_dims]

        p_df = pl.DataFrame(
            np.concatenate([indices, long_post], axis=1),
            schema=([("draw", pl.Int64)] + dim_names + [(v, pl.Float64)]),
        )

        if i_var == 0:
            df = p_df
        else:
            df = df.join(
                p_df, on=[col for col in df.columns if col in p_df.columns]
            )
        pass

    return df


def plot_posterior(
    var: str,
    draws: pl.DataFrame,
    obs_signal: ArrayLike = None,
    ylab: str = None,
    xlab: str = "Time",
    samples: int = 25,
) -> plt.Figure:
    """
    Plot the posterior distribution of a variable

    Parameters
    ----------
    var : str
        Name of the variable to plot
    model : Model
        Model object
    obs_signal : ArrayLike, optional
        Observed signal to plot as reference
    ylab : str, optional
        Label for the y-axis
    xlab : str, optional
        Label for the x-axis
    samples : int, optional
        Number of samples to plot

    Returns
    -------
    plt.Figure
    """

    if ylab is None:
        ylab = var

    fig, ax = plt.subplots(figsize=[4, 5])

    # Reference signal (if any)
    if obs_signal is not None:
        ax.plot(obs_signal, color="black")

    samp_ids = np.random.randint(size=samples, low=0, high=999)

    for samp_id in samp_ids:
        sub_samps = draws.filter(pl.col("draw") == samp_id).sort(
            pl.col("time")
        )
        ax.plot(
            sub_samps.select("time").to_numpy(),
            sub_samps.select(var).to_numpy(),
            color="darkblue",
            alpha=0.1,
        )

    # Some labels
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    # Adding a legend
    ax.plot([], [], color="darkblue", alpha=0.9, label="Posterior samples")

    if obs_signal is not None:
        ax.plot([], [], color="black", label="Observed signal")

    ax.legend()

    return fig
