# numpydoc ignore=ES01,SA01,EX01
"""
Generate 126-day synthetic CA hospital + ED visit data from known R(t).

This script defines R(t) directly, runs a renewal equation forward,
and convolves with signal-specific delay PMFs to produce two
observation streams.  All true parameters are saved alongside the
synthetic observations so tutorials can demonstrate posterior recovery.

The start date (2023-11-05, Sunday) and length (126 days = 18 weeks)
are chosen so that the daily time axis aligns exactly with 18 complete
MMWR epiweeks, with no days trimmed at either end during weekly
aggregation.

Outputs (in pyrenew/datasets/synthetic_CA_126/):
- true_parameters.json
- daily_infections.csv
- daily_ed_visits.csv
- daily_hospital_admissions.csv
- weekly_hospital_admissions.csv

Run from repo root::

    python -m pyrenew.datasets.datagen_he_CA_126

For a separate dataset with Rt constant within each Sunday-Saturday week::

    python -m pyrenew.datasets.datagen_he_CA_126 --rt-cadence weekly

The weekly option writes to ``synthetic_CA_126_weekly_rt/``. It additionally
saves ``weekly_rt.csv``, ``initial_infections.csv``, and
``expected_observations.csv``. Its I0 is new infections per resident on the
first initialization day, matching PopulationInfections. Observation means
are not floored. Weekly generation uses 64-bit arithmetic for reproducibility.
Use ``--output-dir`` and ``--seed`` to generate additional realizations.
The default daily option retains the original dataset generation behavior.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import jax
import numpy as np
import polars as pl

from pyrenew.convolve import compute_delay_ascertained_incidence
from pyrenew.math import r_approx_from_R
from pyrenew.time import daily_to_mmwr_epiweekly, get_sequential_day_of_week_indices

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "pyrenew" / "datasets" / "synthetic_CA_126"
WEEKLY_OUTPUT_DIR = OUTPUT_DIR.with_name("synthetic_CA_126_weekly_rt")

RNG_SEED = 20240101
POPULATION = 39_512_223

GEN_INT_PMF = np.array(
    [0.6326975, 0.2327564, 0.0856263, 0.03150015, 0.01158826, 0.00426308, 0.0015683]
)

HOSP_DELAY_PMF = np.array(
    [
        0.0,
        0.00469384736487552,
        0.0145200073436112,
        0.0278627741704387,
        0.0423656492135518,
        0.0558071445014868,
        0.0665713169684116,
        0.0737925805176124,
        0.0772854627892072,
        0.0773666390616176,
        0.0746515449009949,
        0.0698761436052596,
        0.0637663813017696,
        0.0569581929821651,
        0.0499600186601535,
        0.0431457477049282,
        0.0367662806214045,
        0.0309702535668237,
        0.0258273785539499,
        0.0213504646948306,
        0.0175141661880584,
        0.0142698211023571,
        0.0115565159519833,
        0.00930888979824423,
        0.00746229206759215,
        0.00595605679409682,
        0.00473519993107751,
        0.00375117728281841,
        0.00296198928038098,
        0.00233187862772459,
        0.00183079868293457,
        0.00143377454057296,
        0.00107076258525208,
        0.000773006742366448,
        0.000539573690886396,
        0.000364177599116743,
        0.000237727628685579,
        0.000150157714457011,
        0.0000918283319498657,
        0.0000544079947589854,
        0.0000312548818921465,
        0.0000174202619730274,
        9.42698047424713e-6,
        4.95614149002087e-6,
        2.53275674485913e-6,
        1.25854819834554e-6,
        6.08116579596933e-7,
        2.85572858589747e-7,
        1.30129404249734e-7,
        5.73280599448306e-8,
        2.4219376577964e-8,
        9.6316861194457e-9,
        3.43804936850951e-9,
        9.34806280366888e-10,
        0.0,
    ]
)

ED_DELAY_PMF = np.array(
    [
        0.0,
        0.0213253,
        0.17156943,
        0.23836233,
        0.20200046,
        0.14144434,
        0.09118459,
        0.0567108,
        0.03480426,
        0.0213253,
        0.01312726,
        0.00814594,
    ]
)

IHR = 0.005
IEDR = 0.0075
I0_PER_CAPITA = 5e-4
WEEKLY_I0_PER_CAPITA = 1e-6
NEGBINOM_CONCENTRATION_HOSP = 350.0
NEGBINOM_CONCENTRATION_HOSP_WEEKLY = 100.0
NEGBINOM_CONCENTRATION_ED = 50.0
DOW_EFFECTS = np.array([1.15, 1.12, 1.08, 1.05, 0.98, 0.82, 0.80])

START_DATE = date(2023, 11, 5)
N_INIT = max(50, len(HOSP_DELAY_PMF), len(ED_DELAY_PMF))


def build_true_rt(rt_cadence: Literal["daily", "weekly"] = "daily") -> np.ndarray:
    """
    Build a daily-length R(t) trajectory with daily or weekly changes.

    Phases: decline from 1.2 to 0.8 (63 d), rise from 0.8 to 1.1 (42 d),
    decline from 1.1 to 0.85 (21 d). Total 126 days = 18 weeks.
    Weekly values use the start of each seven-day block of this trajectory.

    Parameters
    ----------
    rt_cadence
        ``"daily"`` for the original piecewise-linear trajectory, or
        ``"weekly"`` to hold each week's starting value for seven days.

    Returns
    -------
    np.ndarray
        R(t) trajectory.

    Raises
    ------
    ValueError
        If the cadence is neither daily nor weekly.
    """
    if rt_cadence not in ("daily", "weekly"):
        raise ValueError("rt_cadence must be 'daily' or 'weekly'")
    segments = [
        (63, 1.2, 0.8),
        (42, 0.8, 1.1),
        (21, 1.1, 0.85),
    ]
    rt = np.concatenate(
        [
            np.linspace(start, end, length, endpoint=False)
            for length, start, end in segments
        ]
    )
    return rt if rt_cadence == "daily" else np.repeat(rt[::7], 7)


def run_renewal(
    rt: np.ndarray,
    gen_int: np.ndarray,
    i0_total: float,
    n_init: int,
    *,
    i0_at_start: bool = False,
) -> np.ndarray:
    """
    Run a discrete renewal equation forward in time.

    Seed infections are placed as an exponentially growing
    trajectory over ``n_init`` days, then the renewal equation
    is applied for ``len(rt)`` days.

    Parameters
    ----------
    rt : np.ndarray
        Effective reproduction number trajectory.
    gen_int : np.ndarray
        Generation interval PMF (sums to 1).
    i0_total : float
        Infection count on the first initialization day when
        ``i0_at_start=True``. Otherwise, the exponential seed curve's
        extrapolated value at day 0, preserving the original convention.
    n_init : int
        Number of seed days before day 0.
    i0_at_start : bool
        Whether i0_total is the count at the beginning of initialization.

    Returns
    -------
    np.ndarray
        Infections of shape ``(n_init + len(rt),)``.
    """
    n_days = len(rt)
    n_total = n_init + n_days
    infections = np.zeros(n_total)

    r0_approx = r_approx_from_R(rt[0], gen_int, 8)
    seed_times = np.arange(n_init) if i0_at_start else np.arange(-n_init, 0)
    infections[:n_init] = i0_total * np.exp(r0_approx * seed_times)

    g = gen_int
    g_len = len(g)
    for t in range(n_init, n_total):
        lookback = min(t, g_len)
        convolution = np.sum(infections[t - lookback : t][::-1] * g[:lookback])
        infections[t] = rt[t - n_init] * convolution

    return infections


def apply_day_of_week_effects(
    values: np.ndarray, dow_effects: np.ndarray, first_day_dow: int
) -> np.ndarray:
    """
    Apply multiplicative day-of-week effects to daily values.

    Parameters
    ----------
    values : np.ndarray
        Daily values.
    dow_effects : np.ndarray
        Multiplicative effects for each day (length 7, ISO convention:
        0 = Monday, 6 = Sunday). Should sum to 7 to preserve weekly totals.
    first_day_dow : int
        ISO day-of-week of the first element in values.

    Returns
    -------
    np.ndarray
        Adjusted daily values.
    """
    day_indices = get_sequential_day_of_week_indices(first_day_dow, len(values))
    return values * dow_effects[day_indices]


def sample_negbinom(
    mu: np.ndarray,
    concentration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample from NegativeBinomial2 parameterization.

    Parameters
    ----------
    mu : np.ndarray
        Mean values (must be non-negative).
    concentration : float
        Concentration parameter (higher = less overdispersion).
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    np.ndarray
        Integer counts.
    """
    p = concentration / (concentration + mu)
    return rng.negative_binomial(n=concentration, p=p)


def build_weekly_hosp_frame(
    weekly_values: np.ndarray,
    start_date: date,
) -> pl.DataFrame:
    """
    Build a weekly hospital admissions frame with MMWR week-ending dates.

    Parameters
    ----------
    weekly_values : np.ndarray
        One value per MMWR epiweek (aggregated by the caller).
    start_date : date
        Date of day 0 of the daily time series the weekly values were
        aggregated from. Used to compute the date of the first
        Saturday (week-ending day).

    Returns
    -------
    pl.DataFrame
        Columns: week_end, weekly_hosp_admits.
    """
    first_day_dow = start_date.weekday()
    days_to_first_sunday = (6 - first_day_dow) % 7
    first_week_end = start_date + timedelta(days=days_to_first_sunday + 6)
    n_weeks = len(weekly_values)
    week_ends = [first_week_end + timedelta(weeks=i) for i in range(n_weeks)]
    return pl.DataFrame(
        {
            "week_end": week_ends,
            "weekly_hosp_admits": np.asarray(weekly_values).tolist(),
        }
    )


def generate(
    *,
    rt_cadence: Literal["daily", "weekly"] = "daily",
    output_dir: Path | None = None,
    seed: int = RNG_SEED,
) -> None:
    """
    Write synthetic observations and their generating parameters to disk.

    Parameters
    ----------
    rt_cadence
        Frequency of Rt changes. The default retains the original daily
        dataset; ``"weekly"`` generates constant Rt within each calendar week.
    output_dir
        Destination directory. Defaults to ``synthetic_CA_126`` for daily
        Rt and ``synthetic_CA_126_weekly_rt`` for weekly Rt.
    seed
        Random seed for observation noise. Rt and expected counts are fixed.
    """
    if output_dir is None:
        output_dir = WEEKLY_OUTPUT_DIR if rt_cadence == "weekly" else OUTPUT_DIR
    with jax.enable_x64(rt_cadence == "weekly" or jax.config.x64_enabled):
        _generate(rt_cadence, output_dir, seed)


def _generate(
    rt_cadence: Literal["daily", "weekly"], output_dir: Path, seed: int
) -> None:
    """Generate and save observations at the selected arithmetic precision."""
    true_rt = build_true_rt(rt_cadence)
    weekly_rt = rt_cadence == "weekly"
    gen_int, hosp_delay, ed_delay = [
        pmf / pmf.sum() if weekly_rt else pmf
        for pmf in (GEN_INT_PMF, HOSP_DELAY_PMF, ED_DELAY_PMF)
    ]
    n_init = (
        max(len(GEN_INT_PMF), len(HOSP_DELAY_PMF) - 1, len(ED_DELAY_PMF) - 1)
        if weekly_rt
        else N_INIT
    )
    i0_per_capita = WEEKLY_I0_PER_CAPITA if weekly_rt else I0_PER_CAPITA
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_days = len(true_rt)
    i0_total = i0_per_capita * POPULATION

    infections_full = run_renewal(
        true_rt, gen_int, i0_total, n_init, i0_at_start=weekly_rt
    )
    infections_obs = infections_full[n_init:]

    obs_dates = [START_DATE + timedelta(days=i) for i in range(n_days)]
    first_day_dow = START_DATE.weekday()

    expected_hosp_daily, _ = compute_delay_ascertained_incidence(
        latent_incidence=infections_full,
        delay_incidence_to_observation_pmf=hosp_delay,
        p_observed_given_incident=IHR,
        pad=True,
    )
    expected_hosp_daily = expected_hosp_daily[n_init:]
    if not weekly_rt:
        expected_hosp_daily = np.maximum(expected_hosp_daily, 1.0)
    hosp_daily_obs = sample_negbinom(
        expected_hosp_daily, NEGBINOM_CONCENTRATION_HOSP, rng
    )

    # Weekly hospital admissions: aggregate daily *expected* values to MMWR
    # epiweeks, then apply one NegBin at the weekly scale. This matches the
    # generative assumption of PopulationCounts(aggregation_period=7) with a
    # single NegativeBinomialNoise at the reporting cadence.
    expected_hosp_weekly = np.asarray(
        daily_to_mmwr_epiweekly(expected_hosp_daily, input_data_first_dow=first_day_dow)
    )
    if not weekly_rt:
        expected_hosp_weekly = np.maximum(expected_hosp_weekly, 1.0)
    hosp_weekly_obs = sample_negbinom(
        expected_hosp_weekly, NEGBINOM_CONCENTRATION_HOSP_WEEKLY, rng
    )
    weekly_hosp = build_weekly_hosp_frame(hosp_weekly_obs, START_DATE)

    hosp_daily_df = pl.DataFrame(
        {
            "date": obs_dates,
            "geo_value": ["CA"] * n_days,
            "daily_hosp_admits": hosp_daily_obs.tolist(),
            "pop": [POPULATION] * n_days,
        }
    )
    hosp_daily_df.write_csv(output_dir / "daily_hospital_admissions.csv")

    expected_ed, _ = compute_delay_ascertained_incidence(
        latent_incidence=infections_full,
        delay_incidence_to_observation_pmf=ed_delay,
        p_observed_given_incident=IEDR,
        pad=True,
    )
    expected_ed = expected_ed[n_init:]
    expected_ed = apply_day_of_week_effects(expected_ed, DOW_EFFECTS, first_day_dow)
    if not weekly_rt:
        expected_ed = np.maximum(expected_ed, 1.0)
    ed_obs = sample_negbinom(expected_ed, NEGBINOM_CONCENTRATION_ED, rng)

    infections_df = pl.DataFrame(
        {
            "date": obs_dates,
            "true_infections": infections_obs.tolist(),
            "true_rt": true_rt.tolist(),
        }
    )
    infections_df.write_csv(output_dir / "daily_infections.csv")

    ed_df = pl.DataFrame(
        {
            "date": obs_dates,
            "geo_value": ["CA"] * n_days,
            "disease": ["COVID-19"] * n_days,
            "ed_visits": ed_obs.tolist(),
        }
    )
    ed_df.write_csv(output_dir / "daily_ed_visits.csv")

    weekly_hosp = weekly_hosp.with_columns(
        pl.lit("CA").alias("location"),
        pl.lit(POPULATION).alias("pop"),
    )
    if weekly_rt:
        weekly_hosp = weekly_hosp.with_columns(
            pl.Series("expected_weekly_hosp_admits", expected_hosp_weekly)
        )
    weekly_hosp.write_csv(output_dir / "weekly_hospital_admissions.csv")

    true_params: dict[str, Any] = {
        "description": (
            "True parameters used to generate synthetic 120-day CA data. "
            "All values are known ground truth for posterior recovery checks."
        ),
        "population": POPULATION,
        "start_date": str(START_DATE),
        "n_days": n_days,
        "n_init": n_init,
        "rng_seed": seed,
        "generation_interval_pmf": gen_int.tolist(),
        "i0_per_capita": i0_per_capita,
        "rt_trajectory": {
            "phase_1": {"days": 63, "start": 1.2, "end": 0.8},
            "phase_2": {"days": 42, "start": 0.8, "end": 1.1},
            "phase_3": {"days": 21, "start": 1.1, "end": 0.85},
        },
        "hospitalizations": {
            "ihr": IHR,
            "delay_pmf_source": "infection_admission_interval.tsv",
            "negbinom_concentration_daily": NEGBINOM_CONCENTRATION_HOSP,
            "negbinom_concentration_weekly": NEGBINOM_CONCENTRATION_HOSP_WEEKLY,
            "temporal_resolutions": ["daily", "weekly_epiweek"],
        },
        "ed_visits": {
            "iedr": IEDR,
            "delay_pmf": ed_delay.tolist(),
            "negbinom_concentration": NEGBINOM_CONCENTRATION_ED,
            "day_of_week_effects": DOW_EFFECTS.tolist(),
            "temporal_resolution": "daily",
        },
    }
    if weekly_rt:
        true_params.update(
            description="Known weekly Rt and observations for 126-day posterior recovery tests.",
            rt_cadence="weekly",
            week_start_dow=START_DATE.weekday(),
            initialization_start_date=str(START_DATE - timedelta(days=n_init)),
            i0_definition="New infections per resident on the first initialization day.",
            log_rt_time_0=float(np.log(true_rt[0])),
            rt_trajectory={"weekly_values": true_rt[::7].tolist()},
        )
        true_params["hospitalizations"]["delay_pmf"] = hosp_delay.tolist()
        pl.DataFrame(
            {
                "week_start": obs_dates[::7],
                "week_end": obs_dates[6::7],
                "true_rt": true_rt[::7],
            }
        ).write_csv(output_dir / "weekly_rt.csv")
        pl.DataFrame(
            {
                "date": [START_DATE + timedelta(days=i) for i in range(-n_init, 0)],
                "true_infections": infections_full[:n_init],
            }
        ).write_csv(output_dir / "initial_infections.csv")
        pl.DataFrame(
            {
                "date": obs_dates,
                "expected_hosp_admits": np.asarray(expected_hosp_daily),
                "expected_ed_visits": np.asarray(expected_ed),
            }
        ).write_csv(output_dir / "expected_observations.csv")
    with open(output_dir / "true_parameters.json", "w") as f:
        json.dump(true_params, f, indent=2)
        f.write("\n")

    n_weeks = len(weekly_hosp)
    mean_daily_hosp = float(np.mean(hosp_daily_obs))
    mean_weekly_hosp = float(weekly_hosp["weekly_hosp_admits"].mean())
    mean_daily_ed = float(np.mean(ed_obs))
    print(f"Wrote {n_days} daily infection rows")
    print(f"Wrote {n_days} daily hospital rows (mean {mean_daily_hosp:.0f}/day)")
    print(f"Wrote {n_days} daily ED visit rows (mean {mean_daily_ed:.0f}/day)")
    print(f"Wrote {n_weeks} weekly hospital rows (mean {mean_weekly_hosp:.0f}/week)")
    print(f"Weekly ED/hosp ratio: {mean_daily_ed * 7 / mean_weekly_hosp:.2f}")
    print(f"Output directory: {output_dir}")


def main() -> None:
    """Generate the selected dataset from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rt-cadence",
        choices=("daily", "weekly"),
        default="daily",
        help="Frequency of Rt changes (default: daily).",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Override the dataset output directory."
    )
    parser.add_argument(
        "--seed", type=int, default=RNG_SEED, help="Random seed for observation noise."
    )
    args = parser.parse_args()
    generate(rt_cadence=args.rt_cadence, output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
