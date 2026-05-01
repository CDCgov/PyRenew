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
    Accessor for one signal's ascertainment value.

    Users usually do not instantiate this class directly. It is returned by
    ``AscertainmentModel.for_signal(...)`` and passed to an observation process
    as ``ascertainment_rate_rv``. During model execution, the parent
    ``AscertainmentModel`` samples the actual rate once, and this accessor
    retrieves the signal-specific value without creating additional NumPyro
    sample sites.
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
    Base class for shared ascertainment structure.

    An ascertainment rate is the probability that latent incidence is observed
    in a particular data stream. Examples include an infection-hospitalization
    ratio for hospital admissions or an infection-ED-visit ratio for emergency
    department visits.

    ``AscertainmentModel`` objects make shared structure explicit in a model
    specification. A user defines the shared model once, registers it with
    ``PyrenewBuilder.add_ascertainment(...)``, and passes signal-specific
    accessors into observation processes:

    ```python
    ascertainment = JointAscertainment(...)
    builder.add_ascertainment(ascertainment)

    PopulationCounts(
        name="hospital",
        ascertainment_rate_rv=ascertainment.for_signal("hospital"),
        ...
    )
    ```

    Subclasses own any NumPyro sites needed for the shared structure.
    Accessors returned by ``for_signal()`` read the sampled values from the
    active model context and do not sample independently.
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
        Return an observation-process accessor for one signal.

        Parameters
        ----------
        signal_name
            Name of the signal produced by this ascertainment model. This name
            should match the signal name used when the ascertainment model was
            constructed. It does not have to match the observation process name,
            but using the same name usually makes model specifications easier
            to read.

        Returns
        -------
        AscertainmentSignal
            RandomVariable-compatible accessor for the signal's sampled
            ascertainment rate.

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
        Sample all signal-specific ascertainment values owned by this model.

        Parameters
        ----------
        **kwargs
            Additional model-context arguments supplied by ``MultiSignalModel``.
            Subclasses may ignore unused values.

        Returns
        -------
        Mapping[str, ArrayLike]
            Mapping from signal name to sampled ascertainment rate.
        """
        pass
