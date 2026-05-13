"""
Export PyRenew synthetic H+E data to a production-style HEW model directory.

This module does not import or depend on ``pyrenew-multisignal`` or
``cfa-stf-routine-forecasting``. It only writes the file layout those
repositories expect for a direct model fit.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from textwrap import dedent

import numpy as np

from pyrenew.datasets.infection_admission_interval import (
    load_example_infection_admission_interval,
)
from pyrenew.datasets.synthetic_data import (
    load_synthetic_daily_ed_visits,
    load_synthetic_true_parameters,
    load_synthetic_weekly_hospital_admissions,
)


DEFAULT_RIGHT_TRUNCATION_PMF = [1.0]
"""Default PMF used when synthetic ED reports are treated as complete."""


def _as_float_list(values) -> list[float]:
    """
    Convert an array-like object to a JSON-serializable list of floats.
    """
    return [float(x) for x in values]


def _lognormal_moment_match(pmf: list[float]) -> tuple[float, float]:
    """
    Approximate lognormal parameters for a discrete delay PMF.

    The production HEW model expects a lognormal reference location and scale
    for the inferred hospital delay. The pipeline derives these from the delay
    PMF; here we use a moment-matching approximation over positive day indices.
    """
    probs = np.asarray(pmf, dtype=float)
    probs = probs / probs.sum()
    days = np.arange(probs.size, dtype=float)

    positive = days > 0
    probs = probs[positive]
    days = days[positive]
    probs = probs / probs.sum()

    mean = float(np.sum(days * probs))
    variance = float(np.sum(((days - mean) ** 2) * probs))
    sigma2 = np.log1p(variance / mean**2)
    loc = np.log(mean) - 0.5 * sigma2
    scale = np.sqrt(sigma2)
    return float(loc), float(scale)


def _default_priors_text() -> str:
    """
    Return a minimal priors.py compatible with the production HEW model builder.

    These priors are intended for synthetic-data smoke tests, not production
    forecasting.
    """
    return dedent(
        """
        import jax.numpy as jnp
        import numpyro.distributions as dist
        import pyrenew.transformation as transformation
        from pyrenew.deterministic import DeterministicVariable
        from pyrenew.randomvariable import DistributionalVariable, TransformedVariable

        i0_first_obs_n_rv = DeterministicVariable("i0_first_obs_n_rv", 4 / 10000)
        log_r_mu_intercept_rv = DeterministicVariable(
            "log_r_mu_intercept_rv", jnp.log(1.2)
        )
        eta_sd_rv = DistributionalVariable(
            "eta_sd", dist.TruncatedNormal(0.15, 0.05, low=0)
        )
        autoreg_rt_rv = DistributionalVariable("autoreg_rt", dist.Beta(2, 40))

        inf_feedback_strength_rv = TransformedVariable(
            "inf_feedback",
            DistributionalVariable(
                "inf_feedback_raw",
                dist.LogNormal(jnp.log(50), jnp.log(1.5)),
            ),
            transforms=transformation.AffineTransform(loc=0, scale=-1),
        )

        delay_offset_loc_rv = DistributionalVariable(
            "delay_offset_loc", dist.Normal(0.75, 0.5)
        )
        delay_log_offset_scale_rv = DistributionalVariable(
            "delay_log_offset_scale", dist.Normal(0, 0.5)
        )

        p_ed_visit_mean_rv = DistributionalVariable(
            "p_ed_visit_mean",
            dist.Normal(transformation.SigmoidTransform().inv(0.005), 0.3),
        )
        p_ed_visit_w_sd_rv = DistributionalVariable(
            "p_ed_visit_w_sd_sd", dist.TruncatedNormal(0, 0.01, low=0)
        )
        autoreg_p_ed_visit_rv = DistributionalVariable(
            "autoreg_p_ed_visit_rv", dist.Beta(1, 100)
        )
        ed_visit_wday_effect_rv = TransformedVariable(
            "ed_visit_wday_effect",
            DistributionalVariable(
                "ed_visit_wday_effect_raw",
                dist.Dirichlet(jnp.array([5, 5, 5, 5, 5, 5, 5])),
            ),
            transformation.AffineTransform(loc=0, scale=7),
        )

        ihr_rv = TransformedVariable(
            "ihr",
            DistributionalVariable(
                "logit_ihr",
                dist.Normal(transformation.SigmoidTransform().inv(0.005), 0.3),
            ),
            transforms=transformation.SigmoidTransform(),
        )
        ihr_rel_iedr_rv = DistributionalVariable(
            "ihr_rel_iedr", dist.LogNormal(0, jnp.log(jnp.sqrt(2)))
        )

        ed_neg_bin_concentration_rv = DistributionalVariable(
            "ed_visit_neg_bin_concentration", dist.LogNormal(4, 1)
        )
        hosp_admit_neg_bin_concentration_rv = DistributionalVariable(
            "hosp_admit_neg_bin_concentration", dist.LogNormal(4 + jnp.log(7), 1.5)
        )

        t_peak_rv = DistributionalVariable("t_peak", dist.TruncatedNormal(5, 1, low=0))
        duration_shed_after_peak_rv = DistributionalVariable(
            "durtion_shed_after_peak", dist.TruncatedNormal(12, 3, low=0)
        )
        log10_genome_per_inf_ind_rv = DistributionalVariable(
            "log10_genome_per_inf_ind", dist.Normal(12, 2)
        )
        mode_sigma_ww_site_rv = DistributionalVariable(
            "mode_sigma_ww_site", dist.TruncatedNormal(1, 1, low=0)
        )
        sd_log_sigma_ww_site_rv = DistributionalVariable(
            "sd_log_sigma_ww_site", dist.TruncatedNormal(0, 0.693, low=0)
        )
        mode_sd_ww_site_rv = DistributionalVariable(
            "mode_sd_ww_site", dist.TruncatedNormal(0, 0.25, low=0)
        )

        autoreg_rt_subpop_rv = DistributionalVariable(
            "autoreg_rt_subpop", dist.Beta(1, 4)
        )
        sigma_rt_rv = DistributionalVariable(
            "sigma_rt", dist.TruncatedNormal(0, 0.1, low=0)
        )
        sigma_i_first_obs_rv = DistributionalVariable(
            "sigma_i_first_obs", dist.TruncatedNormal(0, 0.5, low=0)
        )
        offset_ref_logit_i_first_obs_rv = DistributionalVariable(
            "offset_ref_logit_i_first_obs", dist.Normal(0, 0.25)
        )
        offset_ref_log_rt_rv = DistributionalVariable(
            "offset_ref_log_r_t", dist.Normal(0, 0.2)
        )

        ww_ml_produced_per_day = 227000
        max_shed_interval = 26
        """
    ).lstrip()


def write_synthetic_hew_model_dir(
    model_dir: str | Path,
    priors_path: str | Path | None = None,
    overwrite: bool = False,
    location: str = "CA",
    disease: str = "COVID-19",
    right_truncation_offset: int | None = None,
    right_truncation_pmf: list[float] | None = None,
    other_ed_visits_multiplier: float = 10.0,
    delay_pmf_source: str = "ed",
) -> Path:
    """
    Write synthetic H+E data in the production HEW ``model_dir`` layout.

    Parameters
    ----------
    model_dir
        Directory to create. The function writes ``priors.py``,
        ``data/data_for_model_fit.json``, and ``data/model_params.json``.
    priors_path
        Optional path to an external priors file. If omitted, a minimal
        synthetic-data priors file is written.
    overwrite
        Whether to replace an existing ``model_dir``.
    location
        Jurisdiction code to write in the production-shaped data.
    disease
        Disease label to write in the ED data.
    right_truncation_offset
        Offset used by the production ED right-truncation adjustment. Use
        ``None`` for complete synthetic data.
    right_truncation_pmf
        Reporting-delay PMF. Defaults to ``[1.0]`` for complete reports.
    other_ed_visits_multiplier
        Multiplier used to create the production-required
        ``other_ed_visits`` field from disease ED visits. This field is
        included for schema compatibility and is not used by the HEW model.
    delay_pmf_source
        Source for ``inf_to_hosp_admit_pmf`` in ``model_params.json``.
        ``"ed"`` uses the synthetic ED delay PMF. ``"hospital"`` uses the
        example infection-to-admission PMF.

    Returns
    -------
    pathlib.Path
        Path to the created model directory.
    """
    model_dir = Path(model_dir)
    if model_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{model_dir} already exists. Pass overwrite=True to replace it."
            )
        shutil.rmtree(model_dir)

    data_dir = model_dir / "data"
    data_dir.mkdir(parents=True)

    true_params = load_synthetic_true_parameters()
    daily_ed = load_synthetic_daily_ed_visits()
    weekly_hosp = load_synthetic_weekly_hospital_admissions()
    hosp_delay = _as_float_list(
        load_example_infection_admission_interval()["probability_mass"].to_numpy()
    )
    ed_delay = _as_float_list(true_params["ed_visits"]["delay_pmf"])

    if delay_pmf_source == "ed":
        delay_pmf = ed_delay
    elif delay_pmf_source == "hospital":
        delay_pmf = hosp_delay
    else:
        raise ValueError("delay_pmf_source must be one of {'ed', 'hospital'}.")

    delay_loc, delay_scale = _lognormal_moment_match(delay_pmf)
    right_truncation_pmf = (
        DEFAULT_RIGHT_TRUNCATION_PMF
        if right_truncation_pmf is None
        else _as_float_list(right_truncation_pmf)
    )

    ed_visits = daily_ed["ed_visits"].to_list()
    data_for_model_fit = {
        "loc_pop": [int(true_params["population"])],
        "right_truncation_offset": right_truncation_offset,
        "nssp_step_size": 1,
        "nhsn_step_size": 7,
        "nssp_training_data": {
            "date": daily_ed["date"].to_list(),
            "geo_value": [location] * len(daily_ed),
            "observed_ed_visits": [float(x) for x in ed_visits],
            "other_ed_visits": [
                float(max(1.0, round(x * other_ed_visits_multiplier)))
                for x in ed_visits
            ],
            "data_type": ["train"] * len(daily_ed),
        },
        "nssp_training_dates": daily_ed["date"].to_list(),
        "nhsn_training_data": {
            "weekendingdate": weekly_hosp["week_end"].to_list(),
            "jurisdiction": [location] * len(weekly_hosp),
            "hospital_admissions": [
                float(x) for x in weekly_hosp["weekly_hosp_admits"].to_list()
            ],
            "data_type": ["train"] * len(weekly_hosp),
        },
        "nhsn_training_dates": weekly_hosp["week_end"].to_list(),
    }

    model_params = {
        "population_size": int(true_params["population"]),
        "pop_fraction": [1.0],
        "generation_interval_pmf": _as_float_list(
            true_params["generation_interval_pmf"]
        ),
        "right_truncation_pmf": right_truncation_pmf,
        "inf_to_hosp_admit_lognormal_loc": delay_loc,
        "inf_to_hosp_admit_lognormal_scale": delay_scale,
        "inf_to_hosp_admit_pmf": delay_pmf,
    }

    with open(data_dir / "data_for_model_fit.json", "w") as f:
        json.dump(data_for_model_fit, f, indent=2)
    with open(data_dir / "model_params.json", "w") as f:
        json.dump(model_params, f, indent=2)

    if priors_path is None:
        (model_dir / "priors.py").write_text(_default_priors_text())
    else:
        shutil.copyfile(priors_path, model_dir / "priors.py")

    metadata = {
        "source": "pyrenew.datasets.write_synthetic_hew_model_dir",
        "location": location,
        "disease": disease,
        "delay_pmf_source": delay_pmf_source,
        "other_ed_visits_multiplier": other_ed_visits_multiplier,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return model_dir
