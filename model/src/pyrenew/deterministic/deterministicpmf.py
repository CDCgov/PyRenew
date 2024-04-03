# -*- coding: utf-8 -*-

from pyrenew.deterministic.deterministic import DeterministicVariable
from pyrenew.distutil import validate_discrete_dist_vector
from pyrenew.metaclass import RandomVariable


class DeterministicPMF(RandomVariable):
    """A deterministic (degenerate) random variable that represents a PMF."""

    def __init__(
        self,
        vars: tuple,
        label: str = "a_random_variable",
        tol: float = 1e-20,
    ) -> None:
        """Default constructor

        It automatically checks that the elements in `vars` can be indeed
        consireded to be a PMF by calling
        pyrenew.distutil.validate_discrete_dist_vector on each one of its
        entries.

        Parameters
        ----------
        vars : tuple
            A tuple with arraylike objects.
        label : str
            A label to assign to the process.
        tol : float
            Passed to pyrenew.distutil.validate_discrete_dist_vector

        Returns
        -------
        None
        """

        vars2 = list(vars)
        for i in range(0, len(vars2)):
            vars2[i] = validate_discrete_dist_vector(
                discrete_dist=vars2[i],
                tol=tol,
            )

        self.basevar = DeterministicVariable(tuple(vars2), label)

        return None

    @staticmethod
    def validate(vars: tuple) -> None:
        return None

    def sample(
        self,
        random_variables: dict = None,
        constants: dict = None,
    ) -> tuple:
        """Retrieves the deterministic PMF

        Parameters
        ----------

        random_variables : dict
            Ignored. Default None.
        constants : dict
            Ignored. Default None.

        Returns
        -------
        tuple
            Containing the stored values during construction.
        """

        return self.basevar.sample(
            random_variables=random_variables,
            constants=constants,
        )
