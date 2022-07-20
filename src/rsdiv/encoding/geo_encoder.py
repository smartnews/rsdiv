import json
import pkgutil
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
from scipy import spatial

from .base import BaseEncoder


class GeoEncoder(BaseEncoder):
    r"""Plotly Sample Datasets."""
    ECD_PATH: Optional[bytes] = pkgutil.get_data(
        "rsdiv.encoding", "geojson-counties-fips.json"
    )
    if ECD_PATH:
        encode_source: Dict[str, Any] = json.loads(ECD_PATH)

    def __init__(self) -> None:
        super().__init__()
        self.encoder: pd.DataFrame = self.read_source()
        self.coord: List[np.ndarray] = self.encoder.coord.to_list()
        self.index: pd.Index = pd.Index(self.encoder["index"])

    def read_source(self) -> pd.DataFrame:
        geo_county_dict: Dict[str, List] = {}
        for item in self.encode_source["features"]:
            coordinates = item["geometry"]["coordinates"]
            parts = []
            for part in coordinates:
                parts.append(np.asarray(part).squeeze().mean(axis=0).squeeze())
            coord = np.asarray(parts).mean(axis=0)[::-1]  # reverse lat/lng
            name = item["properties"]["NAME"]
            lsad = item["properties"]["LSAD"]
            id = item["id"]
            geo_county_dict[id] = [coord, name, lsad]
        dataframe = pd.DataFrame.from_dict(
            geo_county_dict, orient="index", columns=["coord", "name", "lstd"]
        ).reset_index()
        return dataframe

    def encoding_single(self, org: Union[List, str]) -> Union[int, str]:
        tree = spatial.KDTree(self.coord)
        return str(self.index[int(tree.query(org)[1])])

    def encoding_series(self, series: pd.Series) -> pd.Series:
        encodings = pd.Series(series.apply(lambda x: self.encoding_single(x)))
        return encodings

    def draw_geo_graph(
        self, dataframe: pd.DataFrame, source_name: str, hover_name: str
    ) -> None:
        max_value: float = np.ceil(dataframe[source_name].max())
        min_value: float = np.floor(dataframe[source_name].min())
        fig = px.choropleth(
            dataframe,
            geojson=self.encode_source,
            locations="index",
            color=source_name,
            hover_name=hover_name,
            color_continuous_scale="OrRd",
            range_color=(min_value, max_value),
            scope="usa",
        )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        fig.show()
