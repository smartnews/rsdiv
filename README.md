# Rsdiv: Diversity improvement framework for recommender systems
[![Python](https://img.shields.io/badge/python-3.6%7C3.7%7C3.8%7C3.9-red?logo=Python&logoColor=white)](https://www.python.org)
[![GitHub](https://img.shields.io/github/license/yuanlonghao/reranking?color=blue)](https://github.com/smartnews/rsdiv)

**rsdiv** is a Python package for recommender systems to provide the measurements and improvements for the diversity of results.

Some of its features include:
- various kinds of metrics to measure the diversity of recommender systems from a quantitative view.
- various implementations for diversify algorithms and models.
- benchmarks for comparing and further analysis.
- hyperparameter optimization based on [Optuna](https://github.com/optuna/optuna).

## Basic Usage
### Prepare for a benchmark dataset
Load a benchmark, say, [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/), this is a table benchmark dataset, contains 1 million ratings from 6000 users on 4000 movies.
```
data_loader = MovieLens1MDownLoader()
```
Get the user-item interactions (ratings):
```
ratings = downloader.read_ratings()
```
|    |   userId |   movieId |   rating | timestamp           |
|---:|---------:|----------:|---------:|:--------------------|
|  0 |        1 |      1193 |        5 | 2000-12-31 22:12:40 |
|  1 |        1 |       661 |        3 | 2000-12-31 22:35:09 |
|  2 |        1 |       914 |        3 | 2000-12-31 22:32:48 |
|  ... |        ... |      ... |        ... | ... |
| 1000206 |     6040 |       562 |        5 | 2000-04-25 23:19:06|
| 1000207 |     6040 |      1096 |        4 | 2000-04-26 02:20:48|
| 1000208 |     6040 |      1097 |        4 | 2000-04-26 02:19:29|

Get the users' infomation:
```
users = downloader.read_users()
```
|    |   userId | gender   |   age |   occupation |   zipcode |
|---:|---------:|:---------|------:|-------------:|----------:|
|  0 |        1 | F        |     1 |           10 |     48067 |
|  1 |        2 | M        |    56 |           16 |     70072 |
|  2 |        3 | M        |    25 |           15 |     55117 |
|  ... |        ... | ...        |    ... |     ... |   ... |
| 6037 |     6038 | F        |    56 |            1 |     14706 |
| 6038 |     6039 | F        |    45 |            0 |     01060 |
| 6039 |     6040 | M        |    25 |            6 |     11106 |

Get the items' information:
```
movies = downloader.read_movies()
```
|    |   movieId | title      | genres      |   year |
|---:|----------:|:--------------|:-------|-------:|
|  0 |         1 | Toy Story   | [\'Animation\', "Children\'s", \'Comedy\']  |   1995 |
|  1 |         2 | Jumanji      | [\'Adventure\', "Children\'s", \'Fantasy\'] |   1995 |
|  2 |         3 | Grumpier Old Men | [\'Comedy\', \'Romance\']     |   1995 |
|  ... |   ... | ... | ...     |   ... |
| 3880 | 3950 | Tigerland        | ['Drama'] |   2000 |
| 3881 | 3951 | Two Family House | ['Drama'] |   2000 |
| 3882 | 3952 | Contender, The   | ['Drama', 'Thriller'] |  2000 |


## For developers
make sure you have `pre-commit` installed:
```
pip install pre-commit
pre-commit install
```
