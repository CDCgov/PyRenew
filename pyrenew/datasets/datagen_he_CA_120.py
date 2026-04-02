# numpydoc ignore=ES01,SA01,EX01
"""
Generate 120-day synthetic CA hospital + ED visit data from known R(t).

This script defines R(t) directly, runs a renewal equation forward,
and convolves with signal-specific delay PMFs to produce two
observation streams.  All true parameters are saved alongside the
synthetic observations so tutorials can demonstrate posterior recovery.

Outputs (in pyrenew/datasets/synthetic_CA_120/):
- true_parameters.json
- daily_infections.csv
- daily_ed_visits.csv
- daily_hospital_admissions.csv
- weekly_hospital_admissions.csv

Run from repo root::

    python -m pyrenew.datasets.datagen_he_CA_120
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from pyrenew.convolve import compute_delay_ascertained_incidence
from pyrenew.math import r_approx_from_R
from pyrenew.time import daily_to_mmwr_epiweekly, get_sequential_day_of_week_indices

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "pyrenew" / "datasets" / "synthetic_CA_120"

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
NEGBINOM_CONCENTRATION_HOSP = 350.0
NEGBINOM_CONCENTRATION_ED = 50.0
DOW_EFFECTS = np.array([1.15, 1.12, 1.08, 1.05, 0.98, 0.82, 0.80])

START_DATE = date(2023, 11, 6)
N_INIT = max(50, len(HOSP_DELAY_PMF), len(ED_DELAY_PMF))


def build_true_rt() -> np.ndarray:
    """
    Build a piecewise-linear true R(t) trajectory.

    Phases: decline from 1.2 to 0.8 (60 d), rise from 0.8 to 1.1 (40 d),
    decline from 1.1 to 0.85 (20 d).

    Returns
    -------
    np.ndarray
        R(t) trajectory.
    """
    segments = [
        (60, 1.2, 0.8),
        (40, 0.8, 1.1),
        (20, 1.1, 0.85),
    ]
    rt = np.concatenate(
        [
            np.linspace(start, end, length, endpoint=False)
            for length, start, end in segments
        ]
    )
    n_days = sum([length for length, _, _ in segments])
    return rt


def run_renewal(
    rt: np.ndarray,
    gen_int: np.ndarray,
    i0_total: float,
    n_init: int,
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
        Infections on the last day of the seed period.
    n_init : int
        Number of seed days before day 0.

    Returns
    -------
    np.ndarray
        Infections of shape ``(n_init + len(rt),)``.
    """
    n_days = len(rt)
    n_total = n_init + n_days
    infections = np.zeros(n_total)

    r0_approx = r_approx_from_R(rt[0], gen_int, 8)
    seed_times = np.arange(-n_init, 0)
    infections[:n_init] = i0_total * np.exp(r0_approx * seed_times)

    g = gen_int
    g_len = len(g)
    for t in range(n_init, n_total):
        lookback = min(t, g_len)
        convolution = np.sum(infections[t - lookback : t][::-1] * g[:lookback])
        infections[t] = rt[t - n_init] * convolution

    return infections


def apply_day_of_week_effects(
    values: np.ndarray, dow_effects: np.ndarray, first_dow: int
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
    first_dow : int
        ISO day-of-week of the first element in values.

    Returns
    -------
    np.ndarray
        Adjusted daily values.
    """
    day_indices = get_sequential_day_of_week_indices(first_dow, len(values))
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
        Mean values (must be positive).
    concentration : float
        Concentration parameter (higher = less overdispersion).
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    np.ndarray
        Integer counts.
    """
    mu = np.maximum(mu, 1e-10)
    p = concentration / (concentration + mu)
    return rng.negative_binomial(n=concentration, p=p)


def aggregate_to_epiweeks(
    daily_values: np.ndarray,
    start_date: date,
) -> pl.DataFrame:
    """
    Aggregate daily counts to MMWR epiweek totals (Sun-Sat, labeled by Saturday).

    Only complete 7-day weeks are kept.

    Parameters
    ----------
    daily_values : np.ndarray
        Daily count time series.
    start_date : date
        Date of the first element.

    Returns
    -------
    pl.DataFrame
        Columns: week_end, weekly_hosp_admits.
    """
    first_dow = start_date.weekday()
    weekly_values = daily_to_mmwr_epiweekly(
        np.asarray(daily_values), input_data_first_dow=first_dow
    )
    days_to_first_sunday = (6 - first_dow) % 7
    first_week_end = start_date + timedelta(days=days_to_first_sunday + 6)
    n_weeks = len(weekly_values)
    week_ends = [first_week_end + timedelta(weeks=i) for i in range(n_weeks)]
    return pl.DataFrame(
        {
            "week_end": week_ends,
            "weekly_hosp_admits": np.asarray(weekly_values).tolist(),
        }
    )


def generate() -> None:
    """Generate all synthetic data files and true parameter JSON."""
    rng = np.random.default_rng(RNG_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    true_rt = build_true_rt()
    n_days = len(true_rt)
    i0_total = I0_PER_CAPITA * POPULATION

    infections_full = run_renewal(true_rt, GEN_INT_PMF, i0_total, N_INIT)
    infections_obs = infections_full[N_INIT:]

    obs_dates = [START_DATE + timedelta(days=i) for i in range(n_days)]
    first_dow = START_DATE.weekday()

    expected_hosp_daily, _ = compute_delay_ascertained_incidence(
        latent_incidence=infections_full,
        delay_incidence_to_observation_pmf=HOSP_DELAY_PMF,
        p_observed_given_incident=IHR,
        pad=True,
    )
    expected_hosp_daily = expected_hosp_daily[N_INIT:]
    expected_hosp_daily = np.maximum(expected_hosp_daily, 1.0)
    hosp_daily_obs = sample_negbinom(
        expected_hosp_daily, NEGBINOM_CONCENTRATION_HOSP, rng
    )

    weekly_hosp = aggregate_to_epiweeks(hosp_daily_obs, START_DATE)

    hosp_daily_df = pl.DataFrame(
        {
            "date": obs_dates,
            "geo_value": ["CA"] * n_days,
            "daily_hosp_admits": hosp_daily_obs.tolist(),
            "pop": [POPULATION] * n_days,
        }
    )
    hosp_daily_df.write_csv(OUTPUT_DIR / "daily_hospital_admissions.csv")

    expected_ed, _ = compute_delay_ascertained_incidence(
        latent_incidence=infections_full,
        delay_incidence_to_observation_pmf=ED_DELAY_PMF,
        p_observed_given_incident=IEDR,
        pad=True,
    )
    expected_ed = expected_ed[N_INIT:]
    expected_ed = apply_day_of_week_effects(expected_ed, DOW_EFFECTS, first_dow)
    expected_ed = np.maximum(expected_ed, 1.0)
    ed_obs = sample_negbinom(expected_ed, NEGBINOM_CONCENTRATION_ED, rng)

    infections_df = pl.DataFrame(
        {
            "date": obs_dates,
            "true_infections": infections_obs.tolist(),
            "true_rt": true_rt.tolist(),
        }
    )
    infections_df.write_csv(OUTPUT_DIR / "daily_infections.csv")

    ed_df = pl.DataFrame(
        {
            "date": obs_dates,
            "geo_value": ["CA"] * n_days,
            "disease": ["COVID-19"] * n_days,
            "ed_visits": ed_obs.tolist(),
        }
    )
    ed_df.write_csv(OUTPUT_DIR / "daily_ed_visits.csv")

    weekly_hosp = weekly_hosp.with_columns(
        pl.lit("CA").alias("location"),
        pl.lit(POPULATION).alias("pop"),
    )
    weekly_hosp.write_csv(OUTPUT_DIR / "weekly_hospital_admissions.csv")

    true_params = {
        "description": (
            "True parameters used to generate synthetic 120-day CA data. "
            "All values are known ground truth for posterior recovery checks."
        ),
        "population": POPULATION,
        "start_date": str(START_DATE),
        "n_days": n_days,
        "n_init": N_INIT,
        "rng_seed": RNG_SEED,
        "generation_interval_pmf": GEN_INT_PMF.tolist(),
        "i0_per_capita": I0_PER_CAPITA,
        "rt_trajectory": {
            "phase_1": {"days": 60, "start": 1.2, "end": 0.8},
            "phase_2": {"days": 40, "start": 0.8, "end": 1.1},
            "phase_3": {"days": 20, "start": 1.1, "end": 0.85},
        },
        "hospitalizations": {
            "ihr": IHR,
            "delay_pmf_source": "infection_admission_interval.tsv",
            "negbinom_concentration": NEGBINOM_CONCENTRATION_HOSP,
            "temporal_resolutions": ["daily", "weekly_epiweek"],
        },
        "ed_visits": {
            "iedr": IEDR,
            "delay_pmf": ED_DELAY_PMF.tolist(),
            "negbinom_concentration": NEGBINOM_CONCENTRATION_ED,
            "day_of_week_effects": DOW_EFFECTS.tolist(),
            "temporal_resolution": "daily",
        },
    }
    with open(OUTPUT_DIR / "true_parameters.json", "w") as f:
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
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate()
