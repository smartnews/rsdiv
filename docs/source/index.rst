.. rsdiv documentation master file


rsdiv
====================================

Python package for recommender systems to provide the measurements and improvements for the diversity of results.

This project provides:

* various kinds of metrics to measure the diversity of recommender systems from a quantitative view.
* various implementations for diversifying algorithms and models.
* various implementations of core recommender algorithms.
* benchmarks for comparing and further analysis.
* hyperparameter optimization based on `Optuna <https://github.com/optuna/optuna>`_

Basic Usage
-----------

.. code-block:: python

    import rsdiv as rs

    # prepare for a benchmark dataset
    loader = rs.MovieLens1MDownLoader()

    # evaluate the results in various aspects
    metrics = rs.DiversityMetrics()
    metrics.gini_coefficient(ratings['movieId'])

    # train a recommender
    rc = rs.FMRecommender(ratings, items).fit()

    # improve the diversity
    div = rs.MaximalMarginalRelevance()


.. toctree::
   :maxdepth: 2
   :caption: Contents:

    Installation <installation>
    Tutorial <tutorial_lastfm>
    API Reference <api/index>
    Resources <resources>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
