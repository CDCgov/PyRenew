# numpydoc ignore=ES01,SA01,EX01
"""
Generate 120-day synthetic CA hospital + ED visit data from known R(t).

Unlike generate_ed_data.py (which deconvolves real hospital data),
this script defines R(t) directly, runs a renewal equation forward,
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
from scipy.signal import fftconvolve

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIORS_JSON = REPO_ROOT / "pyrenew" / "datasets" / "he_covid_priors.json"
HOSP_DELAY_TSV = REPO_ROOT / "pyrenew" / "datasets" / "infection_admission_interval.tsv"
OUTPUT_DIR = REPO_ROOT / "pyrenew" / "datasets" / "synthetic_CA_120"

RNG_SEED = 20240101
POPULATION = 39_512_223

GEN_INT_PMF = np.array(
    [0.6326975, 0.2327564, 0.0856263, 0.03150015, 0.01158826, 0.00426308, 0.0015683]
)

IHR = 0.005
IEDR = 0.0075
I0_PER_CAPITA = 5e-4
NEGBINOM_CONCENTRATION_HOSP = 350.0
NEGBINOM_CONCENTRATION_ED = 50.0
DOW_EFFECTS = np.array([1.15, 1.12, 1.08, 1.05, 0.98, 0.82, 0.80])

START_DATE = date(2023, 11, 6)
N_DAYS = 120
N_INIT = 50


def build_true_rt(n_days: int) -> np.ndarray:
    """
    Build a piecewise-linear true R(t) trajectory over ``n_days`` days.

    Phases: decline from 1.2 to 0.8 (60 d), rise from 0.8 to 1.1 (40 d),
    decline from 1.1 to 0.85 (20 d).

    Parameters
    ----------
    n_days : int
        Total number of days.

    Returns
    -------
    np.ndarray
        R(t) trajectory of shape ``(n_days,)``.
    """
    segments = [
        (60, 1.2, 0.8),
        (40, 0.8, 1.1),
        (20, 1.1, 0.85),
    ]
    pieces = []
    for length, start, end in segments:
        pieces.append(np.linspace(start, end, length, endpoint=False))
    rt = np.concatenate(pieces)
    assert len(rt) == n_days, f"Expected {n_days} days, got {len(rt)}"
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
        Effective reproduction number trajectory of shape ``(n_days,)``.
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

    r0_approx = np.log(rt[0]) / np.sum(gen_int * np.arange(len(gen_int)))
    seed_times = np.arange(-n_init, 0)
    infections[:n_init] = i0_total * np.exp(r0_approx * seed_times)

    g = gen_int
    g_len = len(g)
    for t in range(n_init, n_total):
        lookback = min(t, g_len)
        convolution = np.sum(infections[t - lookback : t][::-1] * g[:lookback])
        infections[t] = rt[t - n_init] * convolution

    return infections


def convolve_with_pmf(signal: np.ndarray, pmf: np.ndarray) -> np.ndarray:
    """
    Convolve a signal with a delay PMF, keeping original length.

    Parameters
    ----------
    signal : np.ndarray
        Input time series.
    pmf : np.ndarray
        Delay PMF.

    Returns
    -------
    np.ndarray
        Convolved signal (same length as input).
    """
    return fftconvolve(signal, pmf, mode="full")[: len(signal)]


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
    n = len(values)
    day_indices = (np.arange(n) + first_dow) % 7
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
    dates = [start_date + timedelta(days=int(i)) for i in range(len(daily_values))]
    df = pl.DataFrame({"date": dates, "daily_value": daily_values.tolist()})
    df = df.with_columns(
        pl.col("date")
        .map_elements(
            lambda d: d + timedelta(days=(5 - d.weekday()) % 7),
            return_dtype=pl.Date,
        )
        .alias("week_end")
    )
    weekly = (
        df.group_by("week_end")
        .agg(
            pl.col("daily_value").sum().alias("weekly_hosp_admits"),
            pl.col("date").count().alias("n_days"),
        )
        .filter(pl.col("n_days") == 7)
        .drop("n_days")
        .sort("week_end")
    )
    return weekly


def generate() -> None:
    """Generate all synthetic data files and true parameter JSON."""
    rng = np.random.default_rng(RNG_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(PRIORS_JSON) as f:
        priors = json.load(f)
    ed_delay_pmf = np.array(priors["ed_visits_observation"]["delay_pmf"])
    hosp_delay_pmf = pl.read_csv(HOSP_DELAY_TSV, separator="\t")[
        "probability_mass"
    ].to_numpy()

    true_rt = build_true_rt(N_DAYS)
    i0_total = I0_PER_CAPITA * POPULATION

    infections_full = run_renewal(true_rt, GEN_INT_PMF, i0_total, N_INIT)
    infections_obs = infections_full[N_INIT:]

    obs_dates = [START_DATE + timedelta(days=i) for i in range(N_DAYS)]
    first_dow = START_DATE.weekday()

    expected_hosp_daily = convolve_with_pmf(infections_full, hosp_delay_pmf) * IHR
    expected_hosp_daily = expected_hosp_daily[N_INIT:]
    expected_hosp_daily = np.maximum(expected_hosp_daily, 1.0)
    hosp_daily_obs = sample_negbinom(
        expected_hosp_daily, NEGBINOM_CONCENTRATION_HOSP, rng
    )

    weekly_hosp = aggregate_to_epiweeks(hosp_daily_obs, START_DATE)

    hosp_daily_df = pl.DataFrame(
        {
            "date": obs_dates,
            "geo_value": ["CA"] * N_DAYS,
            "daily_hosp_admits": hosp_daily_obs.tolist(),
            "pop": [POPULATION] * N_DAYS,
        }
    )
    hosp_daily_df.write_csv(OUTPUT_DIR / "daily_hospital_admissions.csv")

    expected_ed = convolve_with_pmf(infections_full, ed_delay_pmf) * IEDR
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
            "geo_value": ["CA"] * N_DAYS,
            "disease": ["COVID-19"] * N_DAYS,
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
        "n_days": N_DAYS,
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
            "delay_pmf": ed_delay_pmf.tolist(),
            "negbinom_concentration": NEGBINOM_CONCENTRATION_ED,
            "day_of_week_effects": DOW_EFFECTS.tolist(),
            "temporal_resolution": "daily",
        },
    }
    with open(OUTPUT_DIR / "true_parameters.json", "w") as f:
        json.dump(true_params, f, indent=2)

    n_weeks = len(weekly_hosp)
    mean_daily_hosp = float(np.mean(hosp_daily_obs))
    mean_weekly_hosp = float(weekly_hosp["weekly_hosp_admits"].mean())
    mean_daily_ed = float(np.mean(ed_obs))
    print(f"Wrote {N_DAYS} daily infection rows")
    print(f"Wrote {N_DAYS} daily hospital rows (mean {mean_daily_hosp:.0f}/day)")
    print(f"Wrote {N_DAYS} daily ED visit rows (mean {mean_daily_ed:.0f}/day)")
    print(f"Wrote {n_weeks} weekly hospital rows (mean {mean_weekly_hosp:.0f}/week)")
    print(f"Weekly ED/hosp ratio: {mean_daily_ed * 7 / mean_weekly_hosp:.2f}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate()
