==============
API References
==============

.. currentmodule:: rsdiv.recommenders

Recommenders
------------
.. autosummary::
    :toctree: api_reference
    :nosignatures:

    BaseRecommender
    FMRecommender
    IALSRecommender

.. currentmodule:: rsdiv.evaluation

Evaluation
----------
.. autosummary::
    :toctree: api_reference
    :nosignatures:

    RankingMetrics
    RankingDistance
    DiversityMetrics
    CosineRelevanceMetric
    InnerProductRelevanceMetric

.. currentmodule:: rsdiv.diversity

Diversity
---------
.. autosummary::
    :toctree: api_reference
    :nosignatures:

    MaximalMarginalRelevance
    SlidingSpectrumDecomposition

.. currentmodule:: rsdiv.dataset

Dataloader
----------
.. autosummary::
    :toctree: api_reference
    :nosignatures:

    BaseDownloader
    MovieLens100KDownLoader
    MovieLens1MDownLoader

.. currentmodule:: rsdiv.embedding

Embedding
---------
.. autosummary::
    :toctree: api_reference
    :nosignatures:

    BaseEmbedder
    FastTextEmbedder

.. currentmodule:: rsdiv.encoding

Encoding
--------
.. autosummary::
    :toctree: api_reference
    :nosignatures:

    BaseEncoder
    GeoEncoder

.. currentmodule:: rsdiv.aggregation

Aggregation
-----------
.. autosummary::
    :toctree: api_reference
    :nosignatures:

    RankProduct
    PMF
