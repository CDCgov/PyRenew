from pyrenew.distutil import validate_discrete_dist_vector
from pyrenew.metaclass import RandomVariable


class DeterministicObs(RandomVariable):
    """A wrapper instance of RandomVariable that allows pass deterministic
    quantities"""

    def __init__(
        self,
        vars: tuple,
        label: str = "a_random_variable",
        validate_pmf: bool = False,
    ) -> None:
        """Default constructor

        Parameters
        ----------
        vars : tuple
            A tuple with arraylike objects.
        label : str
            A label to assign to the process.
        validate_pmf : bool
            When True, it will create of copy of vars and call
            pyrenew.distutil.validate_discrete_dist_vector on each one of its
            entries.

        Returns
        -------
        None
        """

        self.validate(vars)

        if validate_pmf:
            vars2 = list(vars)
            for i in range(0, len(vars2)):
                vars2[i] = validate_discrete_dist_vector(vars2[i])
            self.vars = tuple(vars2)
        else:
            self.vars = vars
        self.label = label

        return None

    def validate(self, vars: tuple) -> None:
        if not isinstance(vars, tuple):
            raise Exception("vars is not a tuple")

        return None

    def sample(
        self,
        random_variables: dict = None,
        constants: dict = None,
    ) -> tuple:
        """Retrieve the value of the deterministic Rv

        Parameters
        ----------
        random_variables : dict
            Ignored.

        constants : dict
            Ignored.

        Returns
        -------
        tuple
            A tuple with the stored arrays during construction.
        """

        return self.vars
