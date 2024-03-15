Using CFA-Multisignal-Renewal
=============================


Example Code Description
------------------------

The function ``pyrenew.sample_infections_rt()`` is used to sample infections:

.. py:function:: pyrenew.sample_infections_rt(I0: ArrayLike, Rt: ArrayLike, reversed_generation_interval_pmf: ArrayLike)

   Sample infections according to a renewal process with a time-varying reproduction number R(t)

   :param I0: Array of initial infections of the same length as the generation interval pmf vector.
   :type I0: ArrayLike
   :param Rt: Timeseries of R(t) values
   :type Rt: ArrayLike
   :param reversed_generation_interval_pmf: Discrete probability mass vector representing the generation interval of the infection process, where the final entry represents an infection 1 time unit in the past, the second-to-last entry represents an infection two time units in the past, etc.
   :type reversed_generation_interval_pmf: ArrayLike
   :return: The timeseries of infections.
   :rtype: Array

.. code-block:: python 

    def sample_infections_rt(
        I0: ArrayLike, Rt: ArrayLike, reversed_generation_interval_pmf: ArrayLike
    ):
    """
    Sample infections according to a
    renewal process with a time-varying
    reproduction number R(t)

    Parameters
    ----------
    I0: ArrayLike
        Array of initial infections of the
        same length as the generation inferval
        pmf vector.

    Rt: ArrayLike
        Timeseries of R(t) values

    reversed_generation_interval_pmf: ArrayLike
        discrete probability mass vector
        representing the generation interval
        of the infection process, where the final
        entry represents an infection 1 time unit in the
        past, the second-to-last entry represents
        an infection two time units in the past, etc.

    Returns
    --------
    The timeseries of infections, as a JAX array
    """
    incidence_func = new_convolve_scanner(reversed_generation_interval_pmf)

    latest, all_infections = jax.lax.scan(incidence_func, I0, Rt)

    return all_infections


.. code-block:: python

   class BasicRenewalModel:
    """
    Implementation of a basic
    renewal model, not abstracted
    or modular, just for testing
    """

    def __init__(
        self,
        Rt0_dist=None,
        Rt_transform=None,
        Rt_rw_dist=None,
        I0_dist=None,
        IHR_dist=None,
        gen_int=None,
        inf_hosp_int=None,
        hosp_observation_model=None,
    ):
        if Rt_transform is None:
            Rt_transform = LogTransform()
        self.Rt_transform = Rt_transform


.. autofunction:: pyrenew.sample_infections_rt