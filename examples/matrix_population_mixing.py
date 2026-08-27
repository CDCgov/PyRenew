"""Demonstrate cross-population infections with matrix-valued R(t)."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotnine as p9

from pyrenew.latent.infection_functions import compute_infections_from_rt

N_DAYS = 60
POPULATIONS = ["Population A", "Population B"]
OUTPUT_DIR = Path(__file__).with_name("matrix_population_mixing_figures")


def simulate(mixing_matrix: jnp.ndarray) -> np.ndarray:
    """
    Run a two-population renewal process with constant mixing.

    Returns
    -------
    np.ndarray
        Daily infections with shape ``(N_DAYS, 2)``.
    """
    generation_interval = jnp.array([0.05, 0.15, 0.30, 0.30, 0.20])
    initial_infections = jnp.zeros((generation_interval.size, 2))
    initial_infections = initial_infections.at[:, 0].set(10.0)
    matrices = jnp.repeat(mixing_matrix[jnp.newaxis], N_DAYS, axis=0)
    return np.asarray(
        compute_infections_from_rt(
            I0=initial_infections,
            Rt=matrices,
            reversed_generation_interval_pmf=generation_interval,
        )
    )


def main() -> None:
    """Simulate disconnected and connected populations and save two figures."""
    disconnected = jnp.array([[1.08, 0.0], [0.0, 0.82]])
    connected = jnp.array([[1.08, 0.0], [0.12, 0.82]])

    trajectories = []
    for scenario, matrix in [
        ("No cross-population transmission", disconnected),
        ("A infects B", connected),
    ]:
        infections = simulate(matrix)
        for population_index, population in enumerate(POPULATIONS):
            trajectories.append(
                pd.DataFrame(
                    {
                        "day": np.arange(N_DAYS),
                        "infections": infections[:, population_index],
                        "population": population,
                        "scenario": scenario,
                    }
                )
            )
    trajectory_data = pd.concat(trajectories, ignore_index=True)

    matrix_data = pd.DataFrame(
        [
            {
                "source": POPULATIONS[source],
                "target": POPULATIONS[target],
                "strength": float(connected[target, source]),
            }
            for target in range(2)
            for source in range(2)
        ]
    )

    trajectory_plot = (
        p9.ggplot(
            trajectory_data,
            p9.aes(x="day", y="infections", color="population"),
        )
        + p9.geom_line(size=1)
        + p9.facet_wrap("scenario", ncol=1, scales="free_y")
        + p9.labs(
            x="Day",
            y="New infections",
            color="Population",
            title="Off-diagonal mixing lets Population A seed Population B",
        )
        + p9.theme_minimal()
        + p9.theme(figure_size=(7, 6), legend_position="top")
    )
    matrix_plot = (
        p9.ggplot(matrix_data, p9.aes(x="source", y="target", fill="strength"))
        + p9.geom_tile(color="white", size=1)
        + p9.geom_text(p9.aes(label="strength"), format_string="{:.2f}", size=12)
        + p9.scale_fill_cmap(name="Transmission", cmap_name="Blues")
        + p9.coord_equal()
        + p9.labs(
            x="Source population",
            y="Target population",
            title="Mixing matrix M (target × source)",
        )
        + p9.theme_minimal()
        + p9.theme(figure_size=(6, 4), legend_position="right")
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    trajectory_plot.save(
        OUTPUT_DIR / "infection_trajectories.png", dpi=150, verbose=False
    )
    matrix_plot.save(OUTPUT_DIR / "mixing_matrix.png", dpi=150, verbose=False)
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
