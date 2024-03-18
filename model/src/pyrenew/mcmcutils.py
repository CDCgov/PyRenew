"""
Utilities to deal with MCMC outputs
"""

import numpy as np
import polars as pl


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
