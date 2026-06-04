"""Declarative driver for benchmark comparison suites.

A suite declares its arms, its :class:`benchmarks.core.comparison.ComparisonSpec`,
and a model builder; this module supplies the CLI, sampler setup, data loading,
the ``--dry-run-data`` short-circuit, and the fit / report loop.

Each arm is built by the shared ``build_fn``, or by a per-arm ``build`` callable
for arms of a different model family.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import numpyro

from benchmarks.core.cli import add_common_args, settings_from_args
from benchmarks.core.comparison import ComparisonSpec
from benchmarks.core.data_source import (
    add_data_source_args,
    load_he_bundle,
    validate_data_source_args,
)
from benchmarks.core.models import BuiltFit
from benchmarks.core.reporting import print_data_summary
from benchmarks.core.run import run_comparison
from benchmarks.core.runner import RT_SITE_NAMES, Candidate
from benchmarks.core.signals import DatasetBundle

BuildFn = Callable[[Any, DatasetBundle], BuiltFit]
ArmsFactory = Callable[[argparse.Namespace, DatasetBundle], "list[Arm]"]


@dataclass(frozen=True)
class Arm:
    """One comparison arm: a named model variant.

    Parameters
    ----------
    name
        Arm name, matching the suite's :class:`ComparisonSpec`.
    config
        Structured configuration passed to the suite's shared build function.
        Ignored when ``build`` is provided.
    config_fields
        Reporting fields used to label and group the candidate. Defaults to the
        flattened ``config`` when it is a dataclass.
    build
        Optional per-arm builder taking a bundle and returning a
        :class:`BuiltFit`, overriding the suite's shared build function for arms
        of a different model family.
    label
        Candidate display name. Defaults to ``name``.
    rt_site_names
        Posterior site names to search for the Rt trajectory, in priority order.
    """

    name: str
    config: Any = None
    config_fields: dict[str, Any] | None = None
    build: Callable[[DatasetBundle], BuiltFit] | None = None
    label: str | None = None
    rt_site_names: tuple[str, ...] = RT_SITE_NAMES

    def to_candidate(
        self, bundle: DatasetBundle, build_fn: BuildFn | None
    ) -> Candidate:
        """Resolve this arm into a runnable :class:`Candidate`.

        Parameters
        ----------
        bundle
            Dataset bundle the candidate fits.
        build_fn
            Shared build function applied to ``config`` when the arm has no
            per-arm ``build``.

        Returns
        -------
        Candidate
            Candidate wrapping this arm's builder, name, and report fields.
        """
        fields = self.config_fields
        if fields is None:
            fields = asdict(self.config) if is_dataclass(self.config) else {}

        if self.build is not None:
            arm_build = self.build

            def candidate_build() -> BuiltFit:
                """Build this arm's model from its per-arm builder.

                Returns
                -------
                BuiltFit
                    Assembled model and run kwargs.
                """
                return arm_build(bundle)
        elif build_fn is not None:
            config = self.config

            def candidate_build() -> BuiltFit:
                """Build this arm's model from the shared build function.

                Returns
                -------
                BuiltFit
                    Assembled model and run kwargs.
                """
                return build_fn(config, bundle)
        else:
            raise ValueError(
                f"Arm {self.name!r} has neither a per-arm build nor a shared build_fn."
            )

        return Candidate(
            name=self.label or self.name,
            arm=self.name,
            config_fields=fields,
            build=candidate_build,
            rt_site_names=self.rt_site_names,
        )


def comparison_suite(
    spec: ComparisonSpec,
    arms: Sequence[Arm] | ArmsFactory,
    build_fn: BuildFn | None = None,
    *,
    description: str | None = None,
    add_args: Callable[[argparse.ArgumentParser], None] | None = None,
    extra_payload: dict[str, Any] | None = None,
) -> Callable[[], None]:
    """Build a ``main()`` entry point for a comparison suite.

    Parameters
    ----------
    spec
        Comparison declaration; its ``name`` is used for output filenames.
    arms
        Either a static list of :class:`Arm`, or a factory called with the
        parsed args and loaded bundle that returns the arms. Use the factory
        form when arms depend on CLI options (e.g. a prior sweep).
    build_fn
        Shared build function applied to each arm's ``config``. Optional when
        every arm carries its own ``build``.
    description
        Help text for the CLI parser, typically the suite module docstring.
    add_args
        Optional hook to register suite-specific CLI arguments.
    extra_payload
        Optional mapping merged into the JSON artifact for provenance.

    Returns
    -------
    Callable[[], None]
        A zero-argument ``main`` that parses args and runs the comparison.
    """

    def main() -> None:
        """Run the suite from the command line."""
        parser = argparse.ArgumentParser(description=description)
        add_data_source_args(parser)
        if add_args is not None:
            add_args(parser)
        add_common_args(parser)
        args = parser.parse_args()
        validate_data_source_args(parser, args)

        settings = settings_from_args(args)
        numpyro.set_host_device_count(settings.num_chains)
        numpyro.enable_x64()

        try:
            bundle = load_he_bundle(args)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}") from exc
        if args.dry_run_data:
            print_data_summary([bundle])
            return

        try:
            arm_list = arms(args, bundle) if callable(arms) else list(arms)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}") from exc
        candidates = [arm.to_candidate(bundle, build_fn) for arm in arm_list]

        run_comparison(
            candidates,
            spec,
            settings,
            comparison_name=spec.name,
            repeats=args.repeats,
            output_dir=None if args.no_write else args.output_dir,
            extra_payload=extra_payload,
        )

    return main
