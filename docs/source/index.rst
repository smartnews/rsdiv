.. rsdiv documentation master file


rsdiv - Reranking for Multi-objective Optimized Recommender Systems
====================================================================

**rsdiv** provides the measurements and improvements for the multi-objective/diversifying tasks.

This project provides:

* various implementations of diversifying/ensemble reranking modules.
* various implementations of core recommender algorithms.
* evaluations for recommender systems from a quantitative/visualize view.
* easy-to-use benchmarks for comparing and further analysis.
* automated hyperparameter optimization.

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
    API Reference <api_reference>
    Resources <resources>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
