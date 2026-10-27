"""Shared initialization and forecast contracts for both infection classes."""

from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import Predictive

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import (
    AR1,
    DifferencedAR1,
    PopulationInfections,
    RandomWalk,
    SubpopulationInfections,
    TemporalProcess,
    WeeklyTemporalProcess,
)
from pyrenew.randomvariable import DistributionalVariable

Parameterization = Literal["state", "innovation"]
ProcessType = type[AR1] | type[DifferencedAR1] | type[RandomWalk]
InfectionType = type[PopulationInfections] | type[SubpopulationInfections]


def make_process(
    process_type: ProcessType,
    parameterization: Parameterization,
    start_dow: int | None = None,
) -> TemporalProcess:
    """
    Return a process with fixed parameters and optional weekly cadence.

    Returns
    -------
    TemporalProcess
        Configured temporal process.
    """
    kwargs: dict[str, Any] = {
        "innovation_sd_rv": DeterministicVariable("sd", 0.1),
        "parameterization": parameterization,
    }
    if process_type is not RandomWalk:
        kwargs["autoreg_rv"] = DeterministicVariable("autoreg", 0.6)
    process = process_type(**kwargs)
    return process if start_dow is None else WeeklyTemporalProcess(process, start_dow)


def make_infections(
    infection_type: InfectionType,
    process_type: ProcessType,
    parameterization: Parameterization,
    n_init: int = 4,
    start_dow: int | None = None,
    deviation_start_dow: int | None = None,
    unequal: bool = False,
    feedback: bool = False,
) -> PopulationInfections | SubpopulationInfections:
    """
    Build a small infection model with an inferred starting log Rt.

    Returns
    -------
    PopulationInfections or SubpopulationInfections
        Configured infection model.
    """
    start = jnp.array([-0.1, 0.2, 0.5]) if unequal else jnp.array(0.2)
    kwargs: dict[str, Any] = {
        "name": "latent",
        "gen_int_rv": DeterministicPMF("gen_int", jnp.array([0.4, 0.6])),
        "n_initialization_points": n_init,
        "I0_rv": DeterministicVariable("I0", 0.001),
        "log_rt_time_0_rv": DistributionalVariable("start", dist.Normal(start, 0.01)),
    }
    if feedback:
        kwargs.update(
            infection_feedback_strength_rv=DeterministicVariable("feedback", -3.0),
            infection_feedback_pmf_rv=kwargs["gen_int_rv"],
        )
    process = make_process(process_type, parameterization, start_dow)
    if infection_type is PopulationInfections:
        return PopulationInfections(single_rt_process=process, **kwargs)
    return SubpopulationInfections(
        baseline_rt_process=process,
        subpop_rt_deviation_process=make_process(
            process_type, parameterization, deviation_start_dow
        ),
        **kwargs,
    )


def fractions(infection_type: InfectionType) -> jax.Array:
    """
    Return fractions for the chosen infection class.

    Returns
    -------
    jax.Array
        Population fractions.
    """
    return (
        jnp.array([1.0])
        if infection_type is PopulationInfections
        else jnp.array([0.2, 0.3, 0.5])
    )


def rt_site(infection_type: InfectionType) -> str:
    """
    Return the daily raw Rt site name.

    Returns
    -------
    str
        Deterministic Rt site name.
    """
    return (
        "latent::rt_single"
        if infection_type is PopulationInfections
        else "latent::rt_subpop"
    )


def sampled_sites(trace: dict[str, Any]) -> dict[str, jax.Array]:
    """
    Extract sampled values, excluding deterministic records and plates.

    Returns
    -------
    dict[str, jax.Array]
        Unobserved sampled values indexed by site name.
    """
    return {
        name: site["value"]
        for name, site in trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    }


@pytest.mark.parametrize(
    "infection_type,unequal",
    [
        (PopulationInfections, False),
        (SubpopulationInfections, False),
        (SubpopulationInfections, True),
    ],
)
@pytest.mark.parametrize("process_type", [RandomWalk, AR1, DifferencedAR1])
@pytest.mark.parametrize("parameterization", ["state", "innovation"])
@pytest.mark.parametrize("n_days", [1, 6])
def test_initialization_and_first_renewal(
    infection_type: InfectionType,
    unequal: bool,
    process_type: ProcessType,
    parameterization: Parameterization,
    n_days: int,
) -> None:
    """Initialization and first renewal share Rt without unused transitions."""
    model = make_infections(
        infection_type, process_type, parameterization, unequal=unequal
    )
    with numpyro.handlers.seed(rng_seed=11), numpyro.handlers.trace() as trace:
        infections = model.sample(
            n_days_post_init=n_days, subpop_fractions=fractions(infection_type)
        )
    rt = trace[rt_site(infection_type)]["value"]
    first_rt = jnp.exp(trace["start"]["value"])
    assert rt.shape == (4 + n_days, len(fractions(infection_type)))
    assert jnp.allclose(rt[:5], first_rt)
    init_name = (
        "latent::I0_init"
        if infection_type is PopulationInfections
        else "latent::I0_init_all_subpops"
    )
    initial = jnp.reshape(trace[init_name]["value"], (4, -1))
    # Solve 0.4 * exp(-r) + 0.6 * exp(-2*r) = 1 / Rt independently.
    growth = -jnp.log((-0.4 + jnp.sqrt(0.16 + 2.4 / first_rt)) / 1.2)
    expected_init = 0.001 * jnp.exp(jnp.arange(4)[:, None] * growth)
    assert jnp.allclose(initial, expected_init, rtol=1e-5)
    expected_first = rt[4] * jnp.sum(
        initial[-2:] * jnp.array([0.6, 0.4])[:, None], axis=0
    )
    assert jnp.allclose(infections.all_subpops[4], expected_first)
    expected_aggregate = jnp.sum(expected_first * fractions(infection_type))
    assert jnp.allclose(
        trace["latent::infections_aggregate"]["value"][4], expected_aggregate
    )
    assert jnp.allclose(trace[rt_site(infection_type) + "_adjusted"]["value"], rt)
    if infection_type is PopulationInfections:
        assert jnp.allclose(rt, jnp.exp(trace["latent::log_rt_single"]["value"]))
    else:
        baseline = trace["latent::log_rt_baseline"]["value"]
        deviations = trace["latent::subpop_deviations"]["value"]
        assert jnp.allclose(deviations.sum(axis=1), 0.0, atol=1e-12)
        assert jnp.allclose(rt, jnp.exp(baseline + deviations))
        assert jnp.allclose(trace["latent::rt_baseline"]["value"], jnp.exp(baseline))
    for name, value in sampled_sites(trace).items():
        if name == "start":
            continue
        expected_length = max(n_days - (2 if process_type is DifferencedAR1 else 1), 0)
        if "init_rate" in name:
            assert value.shape == (rt.shape[1] if name.startswith("subpop") else 1,)
        else:
            assert value.shape[0] == expected_length


@pytest.mark.parametrize(
    "infection_type", [PopulationInfections, SubpopulationInfections]
)
@pytest.mark.parametrize("parameterization", ["state", "innovation"])
def test_feedback_starts_with_renewal(
    infection_type: InfectionType, parameterization: Parameterization
) -> None:
    """Feedback changes first renewal Rt while initialization retains raw Rt."""
    model = make_infections(infection_type, RandomWalk, parameterization, feedback=True)
    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model.sample, rng_seed=4)
    ).get_trace(n_days_post_init=3, subpop_fractions=fractions(infection_type))
    raw = trace[rt_site(infection_type)]["value"]
    adjusted = trace[rt_site(infection_type) + "_adjusted"]["value"]
    assert jnp.allclose(adjusted[:4], raw[:4])
    init_name = (
        "latent::I0_init"
        if infection_type is PopulationInfections
        else "latent::I0_init_all_subpops"
    )
    initial = jnp.reshape(trace[init_name]["value"], (4, -1))
    pressure = jnp.sum(initial[-2:] * jnp.array([0.6, 0.4])[:, None], axis=0)
    assert jnp.allclose(adjusted[4], raw[4] * jnp.exp(-3.0 * pressure))
    assert jnp.all(adjusted[4] < raw[4])


@pytest.mark.parametrize(
    "infection_type", [PopulationInfections, SubpopulationInfections]
)
@pytest.mark.parametrize("parameterization", ["state", "innovation"])
@pytest.mark.parametrize(
    "start_dow,deviation_dow,renewal_dow",
    [(None, None, 2), (0, 0, 0), (6, 6, 2), (0, 6, 3), (6, None, 6), (None, 0, 3)],
)
def test_initialization_length_and_calendar(
    infection_type: InfectionType,
    parameterization: Parameterization,
    start_dow: int | None,
    deviation_dow: int | None,
    renewal_dow: int,
) -> None:
    """Rt depends on the renewal calendar and length, not initialization length."""
    traces = []
    for n_init in (4, 9):
        model = make_infections(
            infection_type,
            RandomWalk,
            parameterization,
            n_init,
            start_dow,
            deviation_dow,
        )
        trace = numpyro.handlers.trace(
            numpyro.handlers.seed(model.sample, rng_seed=7)
        ).get_trace(
            n_days_post_init=15,
            subpop_fractions=fractions(infection_type),
            first_day_dow=(renewal_dow - n_init) % 7,
        )
        traces.append(trace)
        processes = (
            [("log_rt_single", start_dow, "latent::log_rt_single")]
            if infection_type is PopulationInfections
            else [
                ("log_rt_baseline", start_dow, "latent::log_rt_baseline"),
                ("subpop_deviations", deviation_dow, "latent::subpop_deviations"),
            ]
        )
        for prefix, week_start, daily_name in processes:
            daily = trace[daily_name]["value"][n_init:]
            if week_start is None:
                continue
            weekly = trace[prefix + "_weekly"]["value"]
            trim = (renewal_dow - week_start) % 7
            assert weekly.shape[0] == (15 + trim + 6) // 7
            expected = weekly[(jnp.arange(15) + trim) // 7]
            if prefix == "subpop_deviations":
                expected = expected - expected.mean(axis=1, keepdims=True)
            assert jnp.allclose(daily, expected)
            initial = (
                trace["start"]["value"]
                if prefix != "subpop_deviations"
                else jnp.zeros(3)
            )
            direct = numpyro.handlers.condition(
                make_process(RandomWalk, parameterization, week_start).sample,
                data=sampled_sites(trace),
            )
            direct_result = numpyro.handlers.seed(direct, rng_seed=99)(
                n_timepoints=15,
                initial_value=initial,
                n_processes=weekly.shape[1],
                name_prefix=prefix,
                first_day_dow=renewal_dow,
            )
            if prefix == "subpop_deviations":
                direct_result -= direct_result.mean(axis=1, keepdims=True)
            assert jnp.allclose(daily, direct_result)
    assert jnp.array_equal(
        traces[0][rt_site(infection_type)]["value"][4:],
        traces[1][rt_site(infection_type)]["value"][9:],
    )
    for name, value in sampled_sites(traces[0]).items():
        assert jnp.array_equal(value, sampled_sites(traces[1])[name])


@pytest.mark.parametrize(
    "infection_type", [PopulationInfections, SubpopulationInfections]
)
@pytest.mark.parametrize("process_type", [RandomWalk, AR1, DifferencedAR1])
@pytest.mark.parametrize("parameterization", ["state", "innovation"])
@pytest.mark.parametrize("fit_days", [2, 9])
def test_weekly_forecast_boundaries(
    infection_type: InfectionType,
    process_type: ProcessType,
    parameterization: Parameterization,
    fit_days: int,
) -> None:
    """Forecasts replay fitted weeks and create transitions only for new weeks."""
    model = make_infections(
        infection_type,
        process_type,
        parameterization,
        start_dow=6,
        deviation_start_dow=6,
    )
    kwargs = {"subpop_fractions": fractions(infection_type), "first_day_dow": 5}
    # Initialization starts Saturday; renewal starts Wednesday, partway through a week.
    fitted_trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model.sample, rng_seed=9)
    ).get_trace(n_days_post_init=fit_days, **kwargs)
    posterior = {
        name: value[None, ...] for name, value in sampled_sites(fitted_trace).items()
    }
    site = rt_site(infection_type)
    predictive = Predictive(model.sample, posterior_samples=posterior)
    for extension, new_week in ((1, False), (7, True)):
        days = fit_days + extension
        result = predictive(jax.random.key(12), n_days_post_init=days, **kwargs)
        rt = result[site][0]
        assert rt.shape == (4 + days, len(fractions(infection_type)))
        assert result["latent::infections_aggregate"].shape == (1, 4 + days)
        assert jnp.allclose(rt[: 4 + fit_days], fitted_trace[site]["value"])
        assert jnp.allclose(
            result["latent::infections_aggregate"][0, : 4 + fit_days],
            fitted_trace["latent::infections_aggregate"]["value"],
        )
        assert jnp.allclose(rt[:4], rt[4])
        if new_week:
            assert not jnp.allclose(rt[-1], rt[4 + fit_days - 1])
        else:
            assert jnp.allclose(rt[-1], rt[4 + fit_days - 1])
        for prefix in (
            ["log_rt_single"]
            if infection_type is PopulationInfections
            else ["log_rt_baseline", "subpop_deviations"]
        ):
            weekly = result[prefix + "_weekly"][0]
            fitted = fitted_trace[prefix + "_weekly"]["value"]
            assert weekly.shape[0] == fitted.shape[0] + int(new_week)
            assert jnp.allclose(weekly[: len(fitted)], fitted)


@pytest.mark.parametrize(
    "infection_type", [PopulationInfections, SubpopulationInfections]
)
@pytest.mark.parametrize("n_days,weekday", [(0, None), (-1, None), (3, -1), (3, 7)])
def test_invalid_horizon_or_weekday(
    infection_type: InfectionType, n_days: int, weekday: int | None
) -> None:
    """Reject invalid inputs before shifting weekdays or drawing parameters."""
    model = make_infections(infection_type, RandomWalk, "innovation")
    with pytest.raises(ValueError):
        model.sample(
            n_days_post_init=n_days,
            subpop_fractions=fractions(infection_type),
            first_day_dow=weekday,
        )


@pytest.mark.parametrize(
    "infection_type", [PopulationInfections, SubpopulationInfections]
)
def test_weekly_requires_weekday(infection_type: InfectionType) -> None:
    """Weekly Rt still requires a calendar origin."""
    model = make_infections(infection_type, RandomWalk, "innovation", start_dow=6)
    with pytest.raises(ValueError, match="first_day_dow is required"):
        numpyro.handlers.seed(model.sample, rng_seed=1)(
            n_days_post_init=3, subpop_fractions=fractions(infection_type)
        )
