from .base import BaseDownloader
from .movielens_1m import MovieLens1MDownLoader
from .movielens_100k import MovieLens100KDownLoader

__all__ = [
    "BaseDownloader",
    "MovieLens100KDownLoader",
    "MovieLens1MDownLoader",
]
