import pandas as pd
from pathlib import Path
from functools import lru_cache

from .base import BaseDownloader


class MovieLens1MDownLoader(BaseDownloader):
    """MovieLens dataset downLoader for 1M interactions."""

    DOWNLOAD_URL: str = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    DEFAULT_PATH: str = str(Path.cwd() / "ml-1m")

    @lru_cache(maxsize=1)
    def read_ratings(self) -> pd.DataFrame:
        ratings_path: Path = Path(self.DEFAULT_PATH) / "ratings.dat"
        df_ratings: pd.DataFrame = pd.read_csv(
            ratings_path, sep="::", header=None, engine="python"
        )
        df_ratings.columns = pd.Index(["userId", "movieId", "rating", "timestamp"])
        df_ratings["timestamp"] = pd.to_datetime(df_ratings.timestamp, unit="s")
        return df_ratings

    @lru_cache(maxsize=1)
    def read_users(self) -> pd.DataFrame:
        users_path: Path = Path(self.DEFAULT_PATH) / "users.dat"
        df_users: pd.DataFrame = pd.read_csv(
            users_path,
            sep="::",
            header=None,
            engine="python",
            names=["userId", "gender", "age", "occupation", "zipcode"],
        )
        return df_users

    @lru_cache(maxsize=1)
    def read_items(self) -> pd.DataFrame:
        movies_path: Path = Path(self.DEFAULT_PATH) / "movies.dat"
        df_items: pd.DataFrame = pd.read_csv(
            movies_path,
            sep="::",
            header=None,
            encoding="latin-1",
            engine="python",
            names=["itemId", "title", "genres"],
        )
        df_items["release_date"] = df_items["title"].str[-5:-1].astype("int")
        df_items["title"] = df_items["title"].str[:-7]
        df_items["title"] = df_items["title"].apply(
            lambda x: x.split(",")[0] if "," in x else x
        )
        df_items["genres"] = df_items["genres"].apply(lambda x: x.split("|"))
        return df_items
