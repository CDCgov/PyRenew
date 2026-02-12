# numpydoc ignore=GL08

import numpyro

from pyrenew.metaclass import RandomVariable
from pyrenew.transformation import Transform


class TransformedVariable(RandomVariable):
    """
    Class to represent RandomVariables defined
    by taking the output of another RV's
    [`pyrenew.metaclass.RandomVariable.sample`][] method
    and transforming it by a given transformation
    (typically a [`numpyro.distributions.transforms.Transform`][])
    """

    def __init__(
        self,
        name: str,
        base_rv: RandomVariable,
        transforms: Transform | tuple[Transform],
    ):
        """
        Default constructor

        Parameters
        ----------
        name
            A name for the random variable instance.
        base_rv
            The underlying (untransformed) RandomVariable.
        transforms
            Transformation or tuple of transformations
            to apply to the output of
            `base_rv.sample()`; single values will be coerced to
            a length-one tuple. If a tuple, should be the same
            length as the tuple returned by `base_rv.sample()`.

        Returns
        -------
        None
        """
        super().__init__(name=name)
        self.base_rv = base_rv

        if not isinstance(transforms, tuple):
            transforms = (transforms,)
        self.transforms = transforms
        self.validate()

    def sample(self, record=False, **kwargs) -> tuple:
        """
        Sample method. Call self.base_rv.sample()
        and then apply the transforms specified
        in self.transforms.

        Parameters
        ----------
        record
            Whether to record the value of the deterministic
            RandomVariable. Defaults to False.
        **kwargs
            Keyword arguments passed to self.base_rv.sample()

        Returns
        -------
        tuple of the same length as the tuple returned by
        self.base_rv.sample()
        """

        untransformed_values = self.base_rv.sample(**kwargs)

        if not isinstance(untransformed_values, tuple):
            untransformed_values = (untransformed_values,)

        transformed_values = tuple(
            t(uv) for t, uv in zip(self.transforms, untransformed_values)
        )

        if record:
            if len(untransformed_values) == 1:
                numpyro.deterministic(self.name, transformed_values)
            else:
                suffixes = (
                    untransformed_values._fields
                    if hasattr(untransformed_values, "_fields")
                    else range(len(transformed_values))
                )
                for suffix, tv in zip(suffixes, transformed_values):
                    numpyro.deterministic(f"{self.name}_{suffix}", tv)

        if len(transformed_values) == 1:
            transformed_values = transformed_values[0]

        return transformed_values

    def sample_length(self):
        """
        Sample length for a transformed
        random variable must be equal to the
        length of self.transforms or
        validation will fail.

        Returns
        -------
        int
            Equal to the length of `self.transforms`
        """
        return len(self.transforms)

    def validate(self):
        """
        Perform validation checks on a
        TransformedVariable instance,
        confirming that all transformations
        are callable and that the number of
        transformations is equal to the sample
        length of the base random variable.

        Returns
        -------
        None
            on successful validation, or raise a [`ValueError`][]
        """
        for t in self.transforms:
            if not callable(t):
                raise ValueError("All entries in self.transforms must be callable")
        if hasattr(self.base_rv, "sample_length"):
            n_transforms = len(self.transforms)
            n_entries = self.base_rv.sample_length()
            if not n_transforms == n_entries:
                raise ValueError(
                    "There must be exactly as many transformations "
                    "specified as entries self.transforms as there are "
                    "entries in the tuple returned by "
                    "self.base_rv.sample()."
                    f"Got {n_transforms} transforms and {n_entries} "
                    "entries"
                )
