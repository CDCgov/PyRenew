# Developer Documentation

**Note: this document is a work in progress. Contrbitions to all sections are welcome.**

## Github Workflow

- You should request reviews for pull requests from `PyRenew-devs`, unless you have a good reason to request reviews from a smaller group.
- Reviews from all of `PyRenew-devs` are  encouraged, but an approving review from a [codeowner](https://github.com/CDCgov/PyRenew/blob/main/.github/CODEOWNERS) ([@dylanhmorris](https://github.com/dylanhmorris) or [@damonbayer](https://github.com/damonbayer) is required before a pull request can be merged to `main`.
- For CDC contributors: if your pull request has not received a review at the time of the next standup, use standup to find a reviewer.
- External contributors should expect to receive a review within a few days of creating a pull request.
- If you create a draft pull request, indicate what, if anything, about the current pull request should be reviewed.
- Only mark a pull request as “ready for review” if you think it is ready to be merged. This indicates that a thorough, all-encompassing review should be given.

## Installation for Developers

- `poetry install --with dev`
- `pre-commit install`

## Coding Conventions

A variety of coding conventions are enforced by automated tools in continuous integration ([black](https://github.com/psf/black), [isort](https://github.com/PyCQA/isort), [ruff](https://github.com/astral-sh/ruff), [numpydoc-validation](https://github.com/numpy/numpydoc)) via [pre-commit](https://github.com/pre-commit/pre-commit) hooks.

## PyRenew Principles

- Variable naming conventions

  - Use the `data_` prefix for (potentially) observed data.
  - Use the `_rv` suffix for random variables.
  - Use the `observed_` for the output of sample statements where `obs` is a `data_` prefixed object.
  - Thus, code which may reasonably written like `infections = infections.sample(x, obs=infections)` should instead be written `observed_infections = infections_rv.sample(x, obs=data_infections)`.

- Class conventions

  - Composing models is discouraged.
  - Returning anything from `Model.sample` is discouraged. Instead, sample from models using `Predictive` or our `prior_predictive` or `posterior_predictive` functions.
  - Using `numpyro.deterministic` within a `RandomVariable` is discouraged. Only use at the `numpyro.deterministic` `Model` level. If something might need to be recorded from a `RandomVariable`, it should be returned from the `RandomVariable` so it can be recorded at the `Model` level.
  - Using default site names in a `RandomVariable` is discouraged. Only use default site names at the `Model` level.
  - Use `DeterministicVariable`s instead of constants within a model.

- `scan` conventions

  - Use `jax.lax.scan` for any scan whose iterations are deterministic, i.e. iterations contain no internal calls to `RandomVariable.sample()` or `numpyro.sample()`.
  - Use `numpyro.scan` for any scan whose the iterations are stochastic, i.e. the iterations potentially include calls to `RandomVariable.sample()` or `numpyro.sample()`.

- Multidimensional array conventions

  - In a multidimensional array of timeseries, time is always the first dimension. By default, `jax.lax.scan()` and `numpyro.contrib.control_flow.scan()` build output arrays by augmenting the first dimension, and variables are often scanned over time, making default output of scan over time sensible.

## Adding Documentation to Sphinx

The project uses Sphinx to generate documentation. To learn more about Sphinx, visit the [Sphinx documentation](https://www.sphinx-doc.org/en/master/).

### New module or classes
We use Sphinx's `automodule` functionality to generate module API references. When you add a new module, make sure it gets added to the documentation.

For example, under the `./docs/source/pyrenew_reference`, the `index.md` file lists the `HospitalAdmissions` module by adding the following entry:

````markdown
## Hospital Admissions

```{eval-rst}
.. automodule:: pyrenew.latent.hospitaladmissions
   :members:
   :undoc-members:
   :show-inheritance:
```
````

This entry tells Sphinx to generate documentation for the `HospitalAdmissions` module and its members.

### New tutorials

`PyRenew` tutorials are [quarto documents](https://quarto.org) located under [./docs/source/tutorials](https://github.com/CDCgov/PyRenew/tree/main/docs/source/tutorials). Tutorials are automatically rendered using GitHub actions (see the Workflow file [here](https://github.com/CDCgov/PyRenew/actions/workflows/website.yaml)).

To make the new tutorial available in the website, developers should follow these steps:

1. Create a new `quarto` file in the `./docs/source/tutorials` directory. For instance, the `example_with_datasets.qmd` file was added to the repository.
2. Add an entry in the `./docs/source/tutorials/index.md`, for example:

````markdown
```{toctree}
:maxdepth: 2
getting_started
example_with_datasets
```
````

3. Add an `md` entry with the same basename as the `quarto` file in the `./docs/source/tutorials` directory. For instance, the `example_with_datasets.md` file was added to the repository. This last step can be done running the bash script [./hook_scripts/pre-commit-md-placeholder.sh](https://github.com/CDCgov/PyRenew/blob/main/hook_scripts/pre-commit-md-placeholder.sh). Note the script should be executed by `pre-commit`.

### Adding new pages

Sphinx also allows adding arbitrary pages. For instance, all the `PyRenew` tutorials are additional documentation. The steps to add a new page are:

1. Create a `md` file in the appropriate directory. For example, this file about development was added under `./docs/source/developer_documentation.md`.
2. Make sure the new `md` file is included in an indexed file, for instance, `./docs/source/general/ctoc.md`. Here is how it looks:

````markdown
# Complete Table Of Contents

```{toctree}
:maxdepth: 2
../index
../pyrenew_reference/index
../tutorials/index
../genindex
../developer_documentation
```
````

The last entry is the `developer_documentation` page.
