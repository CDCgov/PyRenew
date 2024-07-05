# Multisignal Renewal Project

⚠️ This is a work in progress ⚠️

[![Pre-commit](https://github.com/CDCgov/multisignal-epi-inference/actions/workflows/pre-commit.yaml/badge.svg)](https://github.com/CDCgov/multisignal-epi-inference/actions/workflows/pre-commit.yaml)
[![installation and testing model](https://github.com/CDCgov/multisignal-epi-inference/actions/workflows/model.yaml/badge.svg)](https://github.com/CDCgov/multisignal-epi-inference/actions/workflows/model.yaml)
[![installation and testing pipeline](https://github.com/CDCgov/multisignal-epi-inference/actions/workflows/pipeline.yaml/badge.svg)](https://github.com/CDCgov/multisignal-epi-inference/actions/workflows/pipeline.yaml)
[![Docs: model](https://github.com/CDCgov/multisignal-epi-inference/actions/workflows/website.yaml/badge.svg)](https://github.com/CDCgov/multisignal-epi-inference/actions/workflows/website.yaml)
[![codecov (model)](https://codecov.io/gh/CDCgov/multisignal-epi-inference/graph/badge.svg?token=7Z06HOMYR1)](https://codecov.io/gh/CDCgov/multisignal-epi-inference)

## Overview

The **Multisignal Renewal Project** aims to develop a modeling framework that leverages multiple data sources to enhance CDC's epidemiological modeling capabilities. The project's goal is twofold: (a) **create a Python library** that provides a flexible renewal modeling framework and (b) **develop a pipeline** that leverages this framework to estimate epidemiological parameters from multiple data sources and produce forecasts. The library and pipeline are located in the [**model/**](https://github.com/CDCgov/multisignal-epi-inference/tree/main/model) and [**pipeline/**](https://github.com/CDCgov/multisignal-epi-inference/tree/main/pipeline/) directories of the GitHub repository, respectively.

## Resources

* [The MSR Website](https://cdcgov.github.io/multisignal-epi-inference/tutorials/index.html) provides general documentation and tutorials on using MSR.
* [The Model Equations Sheet](https://github.com/CDCgov/multisignal-epi-inference/blob/main/model/equations.md) describe the mathematics of the renewal processes and models MSR supports.
* Additional reading on renewal processes in epidemiology
  * [_Semi-mechanistic Bayesian modelling of COVID-19 with renewal processes_](https://academic.oup.com/jrsssa/article-pdf/186/4/601/54770289/qnad030.pdf)
  * [_Unifying incidence and prevalence under a time-varying general branching process_](https://link.springer.com/content/pdf/10.1007/s00285-023-01958-w.pdf)

## General Disclaimer

This repository was created for use by CDC programs to collaborate on public health related projects in support of the [CDC mission](https://www.cdc.gov/about/organization/mission.htm).  GitHub is not hosted by the CDC, but is a third party website used by CDC and its partners to share information and collaborate on software. CDC use of GitHub does not imply an endorsement of any one particular service, product, or enterprise.

## Public Domain Standard Notice

This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC § 105. This repository is in
the public domain within the United States, and copyright and related rights in
the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication. By
submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

## License Standard Notice

This repository is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under
the terms of the Apache Software License version 2, or (at your option) any
later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this
program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

## Privacy Standard Notice

This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md)
and [Code of Conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

## Contributing Standard Notice

Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual, irrevocable,
non-exclusive, transferable license to all users under the terms of the
[Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or
later.

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice

This repository is not a source of government records but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).

## Additional Standard Notices

Please refer to [CDC's Template Repository](https://github.com/CDCgov/template)
for more information about [contributing to this repository](https://github.com/CDCgov/template/blob/master/CONTRIBUTING.md),
[public domain notices and disclaimers](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md),
and [code of conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md).
