import pandas as pd
from pathlib import Path
from typing import List
from functools import lru_cache

from .base import BaseDownloader


class MovieLens100KDownLoader(BaseDownloader):
    """MovieLens dataset downLoader for 100K interactions."""

    DOWNLOAD_URL: str = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    DEFAULT_PATH: str = str(Path.cwd() / "ml-100k")

    @lru_cache(maxsize=1)
    def read_ratings(self) -> pd.DataFrame:
        ratings_path: Path = Path(self.DEFAULT_PATH) / "u.data"
        df_ratings: pd.DataFrame = pd.read_csv(
            ratings_path, sep="\t", header=None, engine="python"
        )
        df_ratings.columns = pd.Index(["userId", "movieId", "rating", "timestamp"])
        df_ratings["timestamp"] = pd.to_datetime(df_ratings.timestamp, unit="s")
        return df_ratings

    @lru_cache(maxsize=1)
    def read_users(self) -> pd.DataFrame:
        users_path: Path = Path(self.DEFAULT_PATH) / "u.user"
        df_users: pd.DataFrame = pd.read_csv(
            users_path,
            sep="|",
            header=None,
            engine="python",
            names=["userId", "age", "gender", "occupation", "zipcode"],
        )
        return df_users

    def _read_genres(self) -> List[str]:
        genres_path: Path = Path(self.DEFAULT_PATH) / "u.genre"
        with genres_path.open() as outfile:
            genres = outfile.read()
            return [pair.split("|")[0] for pair in genres.split("\n")][:-2]

    @lru_cache(maxsize=1)
    def read_items(self) -> pd.DataFrame:
        movies_path: Path = Path(self.DEFAULT_PATH) / "u.item"
        genres: List[str] = self._read_genres()
        df_items: pd.DataFrame = pd.read_csv(
            movies_path,
            sep="|",
            header=None,
            encoding="latin-1",
            engine="python",
            names=["itemId", "title", "release_date", "video_release_date", "URL"]
            + genres,
        )
        df_items["title"] = df_items["title"].str[:-7]
        df_items["title"] = df_items["title"].apply(
            lambda x: x.split(",")[0] if "," in x else x
        )
        df_items["release_date"] = pd.to_datetime(df_items.release_date)
        df_items["genres"] = df_items[genres] @ (df_items[genres].columns + "|")
        df_items["genres"] = df_items["genres"].apply(lambda x: x[:-1].split("|"))
        df_items = df_items.drop(columns=genres + ["video_release_date", "URL"])
        return df_items[["itemId", "title", "genres", "release_date"]]
