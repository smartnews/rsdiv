Basic Usage
===========

Prepare for a benchmark dataset
-------------------------------

Load a benchmark, say, `MovieLens 1M Dataset <https://grouplens.org/datasets/movielens/1m/>`_. This is a table benchmark dataset that contains 1 million ratings from 6000 users on 4000 movies.

.. code-block:: python

    import rsdiv as rs
    loader = rs.MovieLens1MDownLoader()

Get the user-item interactions (ratings):

.. code-block:: python

    ratings = loader.read_ratings()

.. csv-table:: Table user-item Interactions
   :file: tables/interactions.csv
   :widths: 20, 20, 20, 10, 30
   :header-rows: 1
