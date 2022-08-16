# Rsdiv: Reranking for Multi-objective Optimized Recommender Systems

[![Python](https://img.shields.io/badge/python3.8%7C3.9-red?logo=Python&logoColor=white)](https://www.python.org)
[![PyPI](https://img.shields.io/pypi/v/rsdiv?color=green)](https://pypi.org/project/rsdiv/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/smartnews/rsdiv)
[![Read the Docs](https://readthedocs.org/projects/rsdiv/badge/?version=latest)](https://rsdiv.readthedocs.io/en/latest/)

**rsdiv** provides the measurements and improvements for the **multi-objective/diversifying** tasks.

Some of its features include:

- various implementations of **diversifying/ensemble** reranking modules.
- various implementations of core **recommender algorithms**.
- evaluations for recommender systems from a **quantitative/visual** view.
- easy-to-use **benchmarks** for comparing and further analysis.
- automated **hyperparameter** optimization.

## Installation

You can simply install the pre-build binaries with:

```bash
pip install rsdiv
```

More installation options can be found [here](https://rsdiv.readthedocs.io/en/latest/installation.html).

## Basic Usage

[Prepare for a benchmark dataset](https://rsdiv.readthedocs.io/en/latest/notebooks/prepare-for-a-benchmark-dataset.html)

[Evaluate the results in various aspects](https://rsdiv.readthedocs.io/en/latest/notebooks/evaluate-the-results-in-various-aspects.html)

[Train and test a recommender](https://rsdiv.readthedocs.io/en/latest/notebooks/train-and-test-a-recommender.html)

[Reranking for diversity improvement](https://rsdiv.readthedocs.io/en/latest/notebooks/reranking-for-diversity.html)

## TODO

### More diversifying algorithms

- implement the Bounded Greedy Selection Strategy, BGS diversify algorithm

- implement the Determinantal Point Process, DPP diversify algorithm

### Hyperparameter optimization

- compatible with [Optuna](https://github.com/optuna/optuna).

### Ensemble ranking

- support the ensemble ranking modules

## For developers

Contributions welcome! Please contact us.

During your development stage, make sure you have `pre-commit` installed in your local environment:

```bash
pip install pre-commit
pre-commit install
```
