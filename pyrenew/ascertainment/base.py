# numpydoc ignore=GL08
"""
Base classes for ascertainment models.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Mapping

from jax.typing import ArrayLike

from pyrenew.ascertainment.context import get_ascertainment_value
from pyrenew.metaclass import RandomVariable


class AscertainmentSignal(RandomVariable):
    """
    Signal-specific accessor for a sampled ascertainment model value.

    This class intentionally creates no NumPyro sites. It reads values sampled
    once by an ``AscertainmentModel`` within the active model execution context.
    """

    def __init__(
        self,
        ascertainment_name: str,
        signal_name: str,
    ) -> None:
        """
        Initialize a signal-specific ascertainment accessor.

        Parameters
        ----------
        ascertainment_name
            Name of the parent ascertainment model.
        signal_name
            Name of the signal to retrieve.
        """
        if not isinstance(ascertainment_name, str) or len(ascertainment_name) == 0:
            raise ValueError(
                "ascertainment_name must be a non-empty string. "
                f"Got {type(ascertainment_name).__name__}: {ascertainment_name!r}"
            )
        if not isinstance(signal_name, str) or len(signal_name) == 0:
            raise ValueError(
                "signal_name must be a non-empty string. "
                f"Got {type(signal_name).__name__}: {signal_name!r}"
            )
        super().__init__(name=f"{ascertainment_name}_{signal_name}")
        self.ascertainment_name = ascertainment_name
        self.signal_name = signal_name

    def sample(self, **kwargs: object) -> ArrayLike:
        """
        Return the sampled ascertainment value for this signal.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments, ignored.

        Returns
        -------
        ArrayLike
            Signal-specific ascertainment value from the active context.
        """
        return get_ascertainment_value(
            ascertainment_name=self.ascertainment_name,
            signal_name=self.signal_name,
        )


class AscertainmentModel(metaclass=ABCMeta):
    """
    Base class for models that produce signal-specific ascertainment rates.

    Ascertainment models own any NumPyro sites needed for shared structure.
    Signal-specific RandomVariables returned by ``for_signal()`` are accessors
    that read already-sampled values.
    """

    def __init__(
        self,
        name: str,
        signals: tuple[str, ...],
    ) -> None:
        """
        Initialize an ascertainment model.

        Parameters
        ----------
        name
            A non-empty string identifying the ascertainment model.
        signals
            Unique signal names produced by this model.
        """
        if not isinstance(name, str) or len(name) == 0:
            raise ValueError(
                f"name must be a non-empty string. Got {type(name).__name__}: {name!r}"
            )
        if not isinstance(signals, tuple) or len(signals) == 0:
            raise ValueError("signals must be a non-empty tuple of strings.")
        if any(not isinstance(signal, str) or len(signal) == 0 for signal in signals):
            raise ValueError("all signals must be non-empty strings.")
        if len(set(signals)) != len(signals):
            raise ValueError("signals must be unique.")

        self.name = name
        self.signals = signals

    def for_signal(self, signal_name: str) -> AscertainmentSignal:
        """
        Return a RandomVariable accessor for a signal-specific rate.

        Parameters
        ----------
        signal_name
            Name of the signal.

        Returns
        -------
        AscertainmentSignal
            RandomVariable accessor for the signal's sampled ascertainment rate.

        Raises
        ------
        ValueError
            If ``signal_name`` is not produced by this model.
        """
        if signal_name not in self.signals:
            raise ValueError(
                f"Unknown signal {signal_name!r} for ascertainment model "
                f"{self.name!r}. Available signals: {self.signals}."
            )
        return AscertainmentSignal(
            ascertainment_name=self.name,
            signal_name=signal_name,
        )

    @abstractmethod
    def sample(self, **kwargs: object) -> Mapping[str, ArrayLike]:
        """
        Sample signal-specific ascertainment rates.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed by the model.

        Returns
        -------
        Mapping[str, ArrayLike]
            Mapping from signal name to sampled ascertainment rate.
        """
        pass
