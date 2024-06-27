# Developer Documentation

## Github Workflow
- You should request reviews for pull requests from `multisignal-epi-inference-devs`, unless you have a good reason to request reviews from a smaller group.
- Reviews from all of `multisignal-epi-inference-devs` are encouraged, but we require an approving review from @dylanhmorris, @damonbayer, or @gvegayon is required before a pull request can be merged to `main`.
- If your pull request has not received a review at the time of the next standup, use standup to find a reviewer.
- If you create a draft pull request, indicate what, if anything, about the current pull request should be reviewed.
- Only mark a pull request as "ready for review" if you think it is ready to be merged. This indicates that a thorough, all-encompassing review should be given.


## Installation for Developers
- poetry install with dev dependencies
- pre-commit install

## Coding Conventions
A variety of coding conventions are enforced by automated tools in continuous integeration (black, isort, ruff, numpydoc-validation)

## PyRenew Principles
- Variable naming conventions
  - Use the `data_` prefix for (potentially) observed data.
  - Use the `_rv` suffix for random variables.
  - Use the `observed_` for the output of sample statements where `obs` is a `data_` prefixed object.
  - Thus, code which may reasonably written like `infections = infections.sample(x, obs=infections)` should instead be written `observed_infections = infections_rv.sample(x, obs=data_infections)`.
- `Model` class conventions
  - Composing models is discouraged.
  - Returning anything from a `Model` is discouraged. Instead, sample from models using `Predictive` or our `prior_predictive` or `posterior_predictive` functions.
  - Using `numpyro.deterministic` within a `RandomVariable` is discouraged. Only use at the `numpyro.deterministic` `Model` level.
  - Using default site names in a `RandomVariable` is discouraged. Only use default site names at the `Model` level.
  - Use `DeterministicVariable`s instead of constants within a model.

## Additional Developer Info
- makefile
- numpydocs
- How does typos work?
- download artifact for website preview
- How to create a new tutorial
- How to add documentation for a new module to the website
-

## Background Information
- renewal papers
- JAX
- Numpyro
