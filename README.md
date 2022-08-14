# Rsdiv: Reranking for Multi-objective Optimized Recommender Systems

[![Python](https://img.shields.io/badge/python3.8%7C3.9-red?logo=Python&logoColor=white)](https://www.python.org)
[![PyPI](https://img.shields.io/pypi/v/rsdiv?color=green)](https://pypi.org/project/rsdiv/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/smartnews/rsdiv)
[![Read the Docs](https://readthedocs.org/projects/rsdiv/badge/?version=latest)](https://rsdiv.readthedocs.io/en/latest/)

**rsdiv** provides the measurements and improvements for the multi-objective/diversifying tasks.

Some of its features include:

- various implementations of diversifying/ensemble reranking modules.
- various implementations of core recommender algorithms.
- evaluations for recommender systems from a quantitative/visual view.
- easy-to-use benchmarks for comparing and further analysis.
- automated hyperparameter optimization.

## Installation

You can simply install the pre-build binaries with:

```bash
pip install rsdiv
```

More installation options can be found [here](https://rsdiv.readthedocs.io/en/latest/installation.html).

## Basic Usage

- [Prepare for a benchmark dataset
](https://rsdiv.readthedocs.io/en/latest/notebooks/prepare-for-a-benchmark-dataset.html)

- [Evaluate the results in various aspects](https://rsdiv.readthedocs.io/en/latest/notebooks/evaluate-the-results-in-various-aspects.html)

- [Train and test a recommender](https://rsdiv.readthedocs.io/en/latest/notebooks/train-and-test-a-recommender.html)

### Train and test a recommender

### Improve the diversity

Not only for categorical labels, but **rsdiv** also supports embedding for items, for example, but the pre-trained 300-dim embedding based on wiki_en by fastText can also be simply imported as:

```python
>>> emb = rs.FastTextEmbedder()
>>> emb.embedding_list(['Comedy', 'Romance'])
>>> array([-0.02061814,  0.06264187,  0.00729847, -0.04322025,  0.04619966, ...])
```

**rsdiv** supports various kinds of diversifying algorithms:

- [Maximal Marginal Relevance](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf), MMR diversify algorithm:

```python
div = rs.MaximalMarginalRelevance()
```

- Modified Gram-Schmidt, MGS diversify algorithm, also known as SSD([Sliding Spectrum Decomposition](https://arxiv.org/pdf/2107.05204.pdf)):

```python
div = rs.SlidingSpectrumDecomposition()
```

The pseudocode codes are:

```python
for i = 1 to n
    v[i] = a[i]
for i = 1 to n
    r[i][i] = ‖v[i]‖
    q[i] = v[i] / r[i][i]
    for j = i+1 to n
        r[i][j] = q[i][*]v[j]
        v[j] = v[j] - r[i][j]q[i]
```

The objective could be formed as:
$\max\limits_{j\in\mathcal{Y}\backslash Y}\left[r_j+\lambda\left||P_{\perp q_j}\right|| \prod\limits_{i\in Y}^{}\left||P_{\perp q_i}\right||\right]$

## TODO

- implement the Bounded Greedy Selection Strategy, BGS diversify algorithm

- implement the Determinantal Point Process, DPP diversify algorithm

### Hyperparameter optimization

- compatible with [Optuna](https://github.com/optuna/optuna).

## For developers

Contributions welcome! Please contact us.

During your development stage, make sure you have `pre-commit` installed in your local environment:

```bash
pip install pre-commit
pre-commit install
```
