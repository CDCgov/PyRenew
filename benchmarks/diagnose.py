"""Single-fit diagnostic harness for the H+E benchmark model.

Builds one :class:`MultiSignalModel` on one dataset under one config and
reports, in order:

- the data-side bundle summary (population, dates, observed value ranges),
- the model-side priors selected and the initial scale they imply,
- prior-predictive ranges for latent infections and predicted observations
  against the observed values,
- whether the initial potential energy and gradient under the sampler's
  ``init_to_sample`` strategy are finite and well scaled,
- optionally, a short NUTS run and its divergence count.

The repro flags force the real-data code path's priors onto the synthetic
bundle one at a time, so the all-divergence real-data failure can be
bisected off the CDC VM. ``--data-source real`` runs the same diagnostics
against a live bundle and requires ``cfa-stf-routine-forecasting``.

Run from the repository root::

    python -m benchmarks.diagnose --all-real
    python -m benchmarks.diagnose --real-i0 --mcmc
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from dataclasses import replace

os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as random  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
from numpyro.infer import init_to_sample  # noqa: E402
from numpyro.infer.util import initialize_model  # noqa: E402

from benchmarks.core.datasets import (  # noqa: E402
    SYNTHETIC_HE_WEEKLY_HOSPITAL,
    SyntheticProvider,
)
from benchmarks.core.models import BuildConfig, BuiltFit, build_he_model  # noqa: E402
from benchmarks.core.signals import DatasetBundle  # noqa: E402

PREDICTED_SITES: tuple[str, ...] = (
    "latent_infections",
    "hospital_predicted",
    "ed_visits_predicted",
)


def _force_real_priors(
    bundle: DatasetBundle,
    *,
    real_i0: bool,
    real_dow: bool,
    real_trunc: bool,
) -> DatasetBundle:
    """Return a synthetic bundle edited to trigger the real-data prior branches.

    Each flag removes a truth value the synthetic bundle carries (or adds the
    right-truncation parameters the real bundle carries), so ``build_he_model``
    selects the same prior it would under ``--data-source real``.

    Parameters
    ----------
    bundle
        Synthetic H+E bundle.
    real_i0
        Drop ``i0_per_capita`` so the vague ``Beta(1, 10)`` I0 prior is used.
    real_dow
        Drop the fixed ED day-of-week effects so the Dirichlet prior is used.
    real_trunc
        Add a right-truncation PMF and offset so that ED right-truncation is
        applied. The synthetic ED delay PMF stands in for the reporting-delay
        PMF, with an offset of 2 matching the default ``n_days_to_omit``.

    Returns
    -------
    DatasetBundle
        Edited bundle.
    """
    fixed_params = dict(bundle.fixed_params)
    signals = dict(bundle.signals)

    if real_i0:
        fixed_params.pop("i0_per_capita", None)

    if real_dow:
        ed = signals["ed_visits"]
        extras = {k: v for k, v in ed.extras.items() if k != "day_of_week_effects"}
        signals["ed_visits"] = replace(ed, extras=extras)

    if real_trunc:
        fixed_params["right_truncation_pmf"] = signals["ed_visits"].extras["delay_pmf"]
        fixed_params["right_truncation_offset"] = 2

    return replace(bundle, fixed_params=fixed_params, signals=signals)


def _load_bundle(args: argparse.Namespace) -> DatasetBundle:
    """Load the synthetic or real H+E bundle, applying any repro flags.

    Returns
    -------
    DatasetBundle
        The bundle the model is built from.
    """
    if args.data_source == "real":
        from benchmarks.core.real_data import RealDataProvider, RealDataSpec

        spec = RealDataSpec(
            disease=args.disease,
            loc_abbr=args.location,
            as_of=args.as_of,
            n_training_days=args.training_days,
            n_days_to_omit=args.omit_last_days,
        )
        return RealDataProvider({"real_he": spec}).get("real_he")

    bundle = SyntheticProvider().get(SYNTHETIC_HE_WEEKLY_HOSPITAL)
    return _force_real_priors(
        bundle,
        real_i0=args.real_i0 or args.all_real,
        real_dow=args.real_dow or args.all_real,
        real_trunc=args.real_trunc or args.all_real,
    )


def _finite(values: jnp.ndarray) -> np.ndarray:
    """Return the finite entries of an array as a flat NumPy array.

    Returns
    -------
    numpy.ndarray
        Finite values only.
    """
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _summarize(values: jnp.ndarray) -> str:
    """Format a min/mean/max summary of the finite entries of an array.

    Returns
    -------
    str
        Compact summary, or a marker when no finite values are present.
    """
    finite = _finite(values)
    if not finite.size:
        return "no finite values"
    return f"min={finite.min():.4g}, mean={finite.mean():.4g}, max={finite.max():.4g}"


def print_data_summary(bundle: DatasetBundle) -> None:
    """Print the data-side summary of a bundle's observations."""
    print("\n=== data summary ===")
    print(f"dataset: {bundle.name}")
    print(f"  population_size: {bundle.population_size:g}")
    print(f"  obs_start_date: {bundle.obs_start_date}")
    print(f"  n_days_post_init: {bundle.n_days_post_init}")
    print(f"  gen_int_pmf_len: {len(bundle.gen_int_pmf)}")
    print(f"  fixed_params: {', '.join(sorted(bundle.fixed_params)) or 'none'}")
    for signal in bundle.signals.values():
        n_missing = int(np.sum(~np.isfinite(np.asarray(signal.values, dtype=float))))
        print(
            f"  signal {signal.name} ({signal.cadence}): n={len(signal.values)}, "
            f"missing={n_missing}, {_summarize(signal.values)}"
        )


def print_model_side_summary(bundle: DatasetBundle) -> None:
    """Print which priors ``build_he_model`` will select and the implied scale.

    Mirrors the branch logic in ``build_he_model`` so the chosen priors are
    visible without rebuilding the model, and reports the initial weekly
    hospital admissions implied by the I0 prior mean for comparison against the
    observed counts.
    """
    print("\n=== model-side priors (as build_he_model will select) ===")
    if "i0_per_capita" in bundle.fixed_params:
        i0_mean = float(bundle.fixed_params["i0_per_capita"])
        print(f"  I0 prior: tight Normal on logit(i0_per_capita={i0_mean:g})")
    else:
        i0_mean = 1.0 / 11.0
        print(f"  I0 prior: real_he_i0_prior() = Beta(1, 10), mean={i0_mean:.4g}")
    ed_extras = bundle.signals["ed_visits"].extras
    if "day_of_week_effects" in ed_extras:
        print("  ED day-of-week: fixed (DeterministicVariable)")
    else:
        print("  ED day-of-week: real_he_ed_day_of_week_prior() = Dirichlet")
    if "right_truncation_pmf" in bundle.fixed_params:
        pmf = np.asarray(bundle.fixed_params["right_truncation_pmf"], dtype=float)
        offset = bundle.fixed_params.get("right_truncation_offset")
        print(
            f"  ED right-truncation: active, pmf_len={pmf.size}, "
            f"pmf_sum={pmf.sum():.4g}, pmf_min={pmf.min():.4g}, offset={offset}"
        )
    else:
        print("  ED right-truncation: inactive")

    baseline_rate = 0.004
    implied_initial = i0_mean * bundle.population_size * baseline_rate * 7.0
    observed = _finite(bundle.signals["hospital"].values)
    print("  --- implied initial scale (I0 prior mean) ---")
    print(f"  initial infections     ~ {i0_mean * bundle.population_size:.4g}")
    print(
        f"  initial weekly hosp    ~ {implied_initial:.4g} "
        f"(i0_mean x pop x baseline_rate={baseline_rate} x 7)"
    )
    if observed.size:
        print(
            f"  observed weekly hosp     {observed.min():.4g} .. {observed.max():.4g} "
            f"(mean {observed.mean():.4g})"
        )
        print(f"  implied / observed-mean  {implied_initial / observed.mean():.4g}x")


def prior_predictive_report(built: BuiltFit, n_draws: int, seed: int) -> None:
    """Run ``n_draws`` seeded forward passes and report predicted vs observed scale.

    Each pass records the deterministic predicted sites (which do not depend on
    the conditioned observations), so the prior-predictive scale of latent
    infections and predicted observations can be compared against the data.
    """
    print(f"\n=== prior predictive ({n_draws} draws) ===")
    model = built.model
    n_init = built.n_initialization_points
    per_draw: dict[str, list[float]] = {name: [] for name in PREDICTED_SITES}
    n_nonfinite = 0
    for i in range(n_draws):
        with numpyro.handlers.seed(rng_seed=seed + i):
            with numpyro.handlers.trace() as trace:
                model.sample(**built.run_kwargs)
        draw_finite = True
        for name in PREDICTED_SITES:
            value = np.asarray(trace[name]["value"], dtype=float)
            if name in ("latent_infections", "ed_visits_predicted"):
                value = value[n_init:]
            finite = value[np.isfinite(value)]
            if finite.size < value.size:
                draw_finite = False
            if finite.size:
                per_draw[name].append(float(finite.mean()))
        if not draw_finite:
            n_nonfinite += 1

    print(
        f"  draws with any non-finite predicted/infection value: "
        f"{n_nonfinite}/{n_draws}"
    )
    for name in PREDICTED_SITES:
        means = np.asarray(per_draw[name], dtype=float)
        if means.size:
            print(
                f"  {name}: per-draw mean median={np.median(means):.4g}, "
                f"range [{means.min():.4g}, {means.max():.4g}]"
            )
        else:
            print(f"  {name}: no finite draws")

    observed = _finite(built.run_kwargs["hospital"]["obs"])
    hosp_means = np.asarray(per_draw["hospital_predicted"], dtype=float)
    if observed.size and hosp_means.size:
        ratio = np.median(hosp_means) / observed.mean()
        print(f"  hospital predicted-mean / observed-mean (median) = {ratio:.4g}x")


def init_finiteness_report(built: BuiltFit, n_seeds: int, seed: int) -> None:
    """Report the initial potential energy and gradient under ``init_to_sample``.

    Matches the kernel's default init strategy. A non-finite potential energy
    or gradient, or a failure to find a valid initial point, indicates the
    density is pathological where the sampler starts, which is the signature of
    uniform divergence.
    """
    print(f"\n=== sampler initialization ({n_seeds} seeds, init_to_sample) ===")
    for i in range(n_seeds):
        rng_key = random.PRNGKey(seed + i)
        try:
            info = initialize_model(
                rng_key,
                built.model.model,
                init_strategy=init_to_sample,
                model_kwargs=built.run_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  seed {seed + i}: initialize_model FAILED: {exc}")
            continue
        pe = float(info.param_info.potential_energy)
        leaves = jax.tree_util.tree_leaves(info.param_info.z_grad)
        grad_norm = float(jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)))
        grad_finite = bool(
            jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))
        )
        print(
            f"  seed {seed + i}: potential_energy={pe:.6g} "
            f"(finite={np.isfinite(pe)}), grad_norm={grad_norm:.6g} "
            f"(finite={grad_finite})"
        )


def short_mcmc_report(built: BuiltFit, seed: int) -> None:
    """Run a short single-chain NUTS fit and report the divergence count."""
    print("\n=== short MCMC (50 warmup, 50 samples, 1 chain) ===")
    built.model.run(
        num_warmup=50,
        num_samples=50,
        rng_key=random.PRNGKey(seed),
        mcmc_args={"num_chains": 1, "progress_bar": False},
        extra_fields=("diverging",),
        **built.run_kwargs,
    )
    extras = built.model.mcmc.get_extra_fields()
    divergences = int(np.sum(np.asarray(extras["diverging"])))
    print(f"  divergences: {divergences}/50")


def _parse_date(arg: str) -> dt.date:
    """Parse a CLI date in YYYY-MM-DD format.

    Returns
    -------
    datetime.date
        Parsed calendar date.
    """
    return dt.date.fromisoformat(arg)


def _parse_args() -> argparse.Namespace:
    """Parse the diagnostic CLI.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-source", choices=("synthetic", "real"), default="synthetic"
    )
    parser.add_argument(
        "--real-i0",
        action="store_true",
        help="Force the real-data Beta(1, 10) I0 prior.",
    )
    parser.add_argument(
        "--real-dow",
        action="store_true",
        help="Force the real-data Dirichlet day-of-week prior.",
    )
    parser.add_argument(
        "--real-trunc",
        action="store_true",
        help="Force ED right-truncation on the synthetic bundle.",
    )
    parser.add_argument(
        "--all-real", action="store_true", help="Apply all three real-data prior swaps."
    )
    parser.add_argument(
        "--parameterization", choices=("innovation", "state"), default="innovation"
    )
    parser.add_argument("--num-draws", type=int, default=50)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mcmc", action="store_true", help="Also run a short single-chain NUTS fit."
    )
    parser.add_argument(
        "--disease", choices=("COVID-19", "Influenza", "RSV"), default="COVID-19"
    )
    parser.add_argument("--location", default="US")
    parser.add_argument("--as-of", type=_parse_date, default=None)
    parser.add_argument("--training-days", type=int, default=150)
    parser.add_argument("--omit-last-days", type=int, default=2)
    args = parser.parse_args()
    if args.data_source == "real" and args.as_of is None:
        parser.error("--as-of is required when --data-source real")
    return args


def main() -> None:
    """Run the single-fit diagnostic from the command line."""
    args = _parse_args()
    numpyro.set_host_device_count(1)
    numpyro.enable_x64()

    bundle = _load_bundle(args)
    print_data_summary(bundle)
    print_model_side_summary(bundle)

    config = BuildConfig(parameterization=args.parameterization)
    built = build_he_model(config, bundle)
    print(
        f"\nbuilt model: n_initialization_points={built.n_initialization_points}, "
        f"config={config}"
    )

    prior_predictive_report(built, args.num_draws, args.seed)
    init_finiteness_report(built, args.num_seeds, args.seed)
    if args.mcmc:
        short_mcmc_report(built, args.seed)


if __name__ == "__main__":
    main()
