Parameterize each ``RandomVariable`` in terms of other ``RandomVariable``\ s
============================================================================

If a ``RandomVariable`` ``Var`` needs to know some numerical parameter
values ``a``, ``b``, etc to perform its ``sample()`` call, those values
should be represented as constiuent ``RandomVariables`` that are
attributes of ``Var`` (i.e. ``Var.a_rv``, ``Var.b_rv``) and are are
``sample()``-d within ``Var.sample()`` (i.e. ``Var.sample()`` gets an
``a`` value by calling ``self.a_rv.sample()`` and a ``b`` value by
calling ``self.b_rv.sample()``).

Often, a value ``a`` should to treated as fixed from the point of view
of ``Var``. Perhaps it is known with near-certainty. Perhaps it is
unknown, but it is sampled/ inferred elsewhere in a larger model. In
that case, use the ``DeterministicVariable()`` construct to pass it to
``Var``, i.e. ``Var.a_rv = DeterministicVariable(fixed_a_value)``. This
corresponds to the `“maximally random” approach <#maximally-random>`__
described in detail below.

Introduction to the problem
---------------------------

Sometimes, a ``RandomVariable`` can only be ``sample()``-d conditional
on one or more other parameter values. As a toy example, imagine writing
a ``RandomVariable`` class to draw from a Normal distribution
conditional on a specified location parameter ``loc`` and a scale
parameter ``scale``.

.. code:: python

   import numpyro
   import numpyro.distributions as dist
   from pyrenew.metaclass import RandomVariable

   class NormalRV(RandomVariable):

       def __init__(self, name: str, loc: float, scale: float):
           self.name = name
           self.loc
           self.scale

       def sample(self, **kwargs):
           return numpyro.sample(
               self.name,
               dist.Normal(loc=self.loc,
                           scale=self.scale),
               **kwargs)


   # instaniate and draw a sample
   my_norm = NormalRV("my_normal_rv", loc=5, scale=0.25)

   with numpyro.handlers.seed(rng_seed=5):
       print(my_norm.sample())

Often, we would like these parameters the ``RandomVariable`` needs to
“know” to be themselves sampled / inferred / stochastic. How should we
set this up? It is useful to characterize two possible extremes: 1.
“Maximally fixed”: From an individual ``RandomVariable`` instance’s
point of view, all the parameters it needs to “know” in order to be
sampled from are fixed. 2. “Maximally random”: From an individual
``RandomVariable`` instance’s point of view, all the parameters it needs
to “know” in order to be sampled from are associated ``RandomVariable``
instances, which it explicitly ``sample()``-s when its own ``sample()``
method is called.

Let’s look at these two extremes in detail, with examples.

Maximally fixed
---------------

Any parameters needed for ``sample()``-ing from a ``RandomVariable`` are
passed as fixed to its constructor. If they are to be random / inferred,
they must be sampled “upstream” and the sampled values passed. Returning
to our toy example, one might do the following to have random ``loc``
and ``scale`` values for a ``NormalRV``:

.. code:: python

   with numpyro.handlers.seed(rng_seed=5):
       random_loc = numpyro.sample("loc", dist.Normal(0, 1))
       random_scale = numpyro.sample("scale", dist.HalfNormal(2))
       my_norm = NormalRV(loc=random_loc, scale=random_scale)
       print(my_norm.sample())

Maximally random
----------------

Any parameters needed for ``sample()``-ing from a ``RandomVariable``
``Var`` are expressed as constiuent ``RandomVariable``\ s of their own
(e.g. ``SubVar1``, ``SubVar2``, etc). When we call ``Var.sample()``, the
constituent random variables are ``sample()``-d “under the hood”
(i.e. ``SubVar1.sample()``, ``SubVar2.sample()``, etc get called within
``Var.sample()``). In this framework, we can express a ``NormalRV`` with
random ``loc`` and ``scale`` values as:

.. code:: python

   from pyrenew.metaclass import DistributionalRV
   class NormalRV():

       def __init__(self, name: str,
                    loc_rv: RandomVariable,
                    scale_rv: RandomVariable):
           self.name = name
           self.loc_rv = loc_rv
           self.scale_rv = scale_rv

       def sample(self, **kwargs):
           loc_sampled_value = self.loc_rv.sample()
           scale_sampled_value = self.scale_rv.sample()
           return numpyro.sample(
               self.name,
               dist.Normal(loc=loc_sampled_value,
                           scale=scale_sampled_value),
               **kwargs)

   loc_rv = DistributionalRV(dist.Normal(0, 1), "loc_dist")
   scale_rv = DistributionalRV(dist.HalfNormal(2), "scale_dist")
   my_norm = NormalRV("my_normal_rv", loc_rv=loc_rv, scale_rv=scale_rv)

   with numpyro.handlers.seed(rng_seed=5):
       print(my_norm.sample())

Why we prefer “Maximally random”
--------------------------------

We believe this approach gives additional flexibility in model building
at minimal cost in terms of additional verbiage/abstraction. Using this
framework, all of the following are valid ``Pyrenew``:

Two ``NormalRV``\ s with distinct inferred scales, with distinct priors,
and distinct inferred ``loc`` values, with distinct priors:

.. code:: python

   scale_rv1 = DistributionalRV(dist.HalfNormal(2), "scale_dist1")
   scale_rv2 = DistributionalRV(dist.HalfNormal(0.5), "scale_dist2")
   loc_rv1 = DistributionalRV(dist.Normal(0, 1), "loc_dist1")
   loc_rv2 = DistributionalRV(dist.Normal(1, 1), "loc_dist2")
   norm1 = NormalRV(loc_rv=loc_rv1, scale_rv=scale_rv1)
   norm2 = NormalRV(loc_rv=loc_rv2, scale_rv=scale_rv2)

Two ``NormalRV``\ s with distinct inferred scales, with a shared prior,
and distinct inferred ``loc``\ s, with distinct priors:

.. code:: python

   scale_rv = DistributionalRV(dist.HalfNormal(0, 2), "scale_dist")
   loc_rv1 = DistributionalRV(dist.Normal(0, 1), "loc_dist1")
   loc_rv2 = DistributionalRV(dist.Normal(1, 1), "loc_dist2")
   norm1 = NormalRV(loc_rv=loc_rv1, scale_rv=scale_rv)
   norm2 = NormalRV(loc_rv=loc_rv2, scale_rv=scale_rv)

Two ``NormalRV``\ s with a single shared inferred ``scale`` and distinct
inferred ``loc``\ s, with distinct priors:

.. code:: python

   scale_rv = DistributionalRV(dist.HalfNormal(2), "scale_dist")
   loc_rv1 = DistributionalRV(dist.Normal(0, 1), "loc1_dist")
   loc_rv2 = DistributionalRV(dist.Normal(1, 1), "loc2_dist")

   # we sample the scale value here explicitly,
   # and the pass it to each of the NormalRVs as a
   # DeterministicVariable
   scale_val = scale_rv.sample()

   shared_scale = DeterministicVariable(scale_val)

   norm1 = NormalRV(loc_rv=loc_rv1, scale_rv=shared_scale)
   norm2 = NormalRV(loc_rv=loc_rv2, scale_rv=shared_scale)

Future possibilities
====================

In the future, we might wish to have ``RandomVariable`` constructors
coerce fixed values to ``DeterministicVariables`` via some
``ensure_rv()`` function, so that this:

.. code:: python

   scale_rv = DistributionalRV(dist.HalfNormal(2), "scale_dist")
   loc_rv1 = DistributionalRV(dist.Normal(0, 1), "loc1_dist")
   loc_rv2 = DistributionalRV(dist.Normal(1, 1), "loc2_dist")
   shared_scale = scale_rv.sample()

   norm1 = NormalRV(loc_rv=loc_rv1, scale_rv=shared_scale)
   norm2 = NormalRV(loc_rv=loc_rv1, scale_rv=shared_scale)

is viable shorthand for this:

.. code:: python

   scale_rv = DistributionalRV(dist.HalfNormal(2), "scale_dist")
   loc_rv1 = DistributionalRV(dist.Normal(0, 1), "loc1_dist")
   loc_rv2 = DistributionalRV(dist.Normal(1, 1), "loc2_dist")

   scale_val = scale_rv.sample()
   shared_scale = DeterministicVariable(scale_val)

   norm1 = NormalRV(loc_rv=loc_rv1, scale_rv=shared_scale)
   norm2 = NormalRV(loc_rv=loc_rv2, scale_rv=shared_scale)
