# Developer Documentation

**Note: this document is a work in progress. Contrbitions to all sections are welcome.**

## GitHub Workflow

- You should request reviews for pull requests from `PyRenew-devs`, unless you have a good reason to request reviews from a smaller group.
- Reviews from all of `PyRenew-devs` are  encouraged, but an approving review from a [codeowner](https://github.com/CDCgov/PyRenew/blob/main/.github/CODEOWNERS) ([@dylanhmorris](https://github.com/dylanhmorris) or [@damonbayer](https://github.com/damonbayer) is required before a pull request can be merged to `main`.
- For CDC contributors: if your pull request has not received a review at the time of the next standup, use standup to find a reviewer.
- External contributors should expect to receive a review within a few days of creating a pull request.
- If you create a draft pull request, indicate what, if anything, about the current pull request should be reviewed.
- Only mark a pull request as “ready for review” if you think it is ready to be merged. This indicates that a thorough, all-encompassing review should be given.

## Installation for Developers

- `uv sync --extra dev`
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

## Documenting code for MkDocs

The project uses [MkDocs](https://www.mkdocs.org/) and [mkdocstrings](https://mkdocstrings.github.io/) to generate documentation.
MkDocs builds the documentation pages from the source files contained in the `docs` directory
and these, in turn, contain `mkdocstrings` directives to include the docstrings in the source code file.

The top-level `Makefile` task `docs` will build the site locally in a new directory `site`

```
make docs
open site/index.html
```

### New module or classes

For each submodule or class, there is a corresponding pages in directory `docs/reference`.

For example, under the `./docs/reference`, the `index.md` file lists the `distributions` module by adding the following entry:

```markdown
# Distributions

::: pyrenew.distributions
```

### New tutorials

`PyRenew` tutorials are [quarto documents](https://quarto.org) located under [./docs/source/tutorials](https://github.com/CDCgov/PyRenew/tree/main/docs/source/tutorials).
The `make docs` Makefile task first renders the `.qmd` files to `.md`, then runs `mkdocs build` to generate the site.

To make the new tutorial available in the website, developers should follow these steps:

1. Create a new `quarto` file in the `./docs/tutorials` directory. For instance, the `example_with_datasets.qmd` file was added to the repository.
2. Add an entry in the `./docs//tutorials/.pages` file to specify the order in which this tutorial  will appear in the navigation sidebar.   The entry specifies the *plain markdown* filename.

For example, if you are adding a tutorial named `seasonal_effects.qmd`, then you would update the
file `docs/tutorials/.pages` as follows

```
arrange:
  - getting_started.md
  - basic_renewal_model.md
  - extending_pyrenew.md
  - hospital_admissions_model.md
  - day_of_the_week.md
  - periodic_effects.md
  - seasonal_effects.md
```



### Adding new pages

To add a new page which is neither source code documentation nor a tutorial:

1. Create a `md` file in the appropriate directory. For example, this file about development was added under `./docs/source/developer_documentation.md`.
2. Make sure the new `md` file is included in the `.pages` file for that directory.

```
nav:
  - reference
  - tutorials
  - developer_documentation.md
```

The last entry is the `developer_documentation` page.
