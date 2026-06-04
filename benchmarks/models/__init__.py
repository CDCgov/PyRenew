"""Reusable benchmark model builders.

This layer sits between :mod:`benchmarks.core` (model-agnostic machinery) and
:mod:`benchmarks.suites` (thin comparison declarations). Each module builds one
model family and returns a :class:`benchmarks.core.models.BuiltFit`. ``he``
builds the hospital + ED-visits family from PyRenew primitives; ``hew`` adapts
the production ``pyrenew-multisignal`` HEW model.
"""
