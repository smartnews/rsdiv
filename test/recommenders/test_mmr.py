from typing import List

import numpy as np
import pandas as pd
import pytest
from pytest import mark, raises

import rsdiv as rs


class TestMaximalMarginalRelevance:
    @pytest.fixture
    def candidate_scale(self) -> int:
        return 200

    @pytest.fixture
    def dataset(self, candidate_scale: int) -> pd.DataFrame:
        loader = rs.MovieLens1MDownLoader()
        dataframe = loader.read_items()
        return dataframe.head(candidate_scale).copy()

    @pytest.fixture
    def embedder(self) -> rs.FastTextEmbedder:
        return rs.FastTextEmbedder()

    @pytest.fixture
    def relevance_scores(self, candidate_scale: int) -> np.ndarray:
        rng = np.random.default_rng(42)
        scores = -np.sort(-rng.random(candidate_scale))
        return scores

    @pytest.fixture
    def similarity_scores(
        self, dataset: pd.DataFrame, embedder: rs.FastTextEmbedder
    ) -> np.ndarray:
        dataframe = dataset
        dataframe["embedding"] = dataframe["genres"].apply(embedder.embedding_list)
        embedding: np.ndarray = np.asarray(list(dataframe["embedding"]))
        scores: np.ndarray = embedding @ embedding.T
        return scores

    @mark.parametrize("lbd", [0.0, 0.2, 0.3])
    @mark.parametrize("k", [50, 70, 100])
    def test_rerank_scale(
        self,
        dataset: pd.DataFrame,
        relevance_scores: np.ndarray,
        similarity_scores: np.ndarray,
        lbd: float,
        k: int,
    ) -> None:
        genres = dataset["genres"]
        mmr = rs.MaximalMarginalRelevance(lbd)
        metrics = rs.DiversityMetrics()
        selected_ind = mmr.rerank(
            relevance_scores, k, similarity_scores=similarity_scores
        )
        gini_org = metrics.gini_coefficient(genres[:k])
        gini_mmr = metrics.gini_coefficient([genres[index] for index in selected_ind])
        assert selected_ind[0] == 0
        assert len(selected_ind) == k
        assert gini_org > gini_mmr

    @mark.parametrize("scale", [150, 160, 170, 180])
    def test_rerank_lambda(
        self,
        dataset: pd.DataFrame,
        relevance_scores: np.ndarray,
        similarity_scores: np.ndarray,
        scale: int,
        lbd_list: List[float] = [0, 0.1, 0.2, 0.3, 1],
    ) -> None:
        genres = dataset["genres"]
        metrics = rs.DiversityMetrics()
        gini_org = metrics.gini_coefficient(genres[:scale])
        gini_ans = []
        for lbd in lbd_list:
            mmr = rs.MaximalMarginalRelevance(lbd)
            selected_ind = mmr.rerank(
                relevance_scores, scale, similarity_scores=similarity_scores
            )
            gini_mmr = metrics.gini_coefficient(
                [genres[index] for index in selected_ind]
            )
            gini_ans.append(gini_mmr)
        assert selected_ind[0] == 0
        assert gini_ans[-1] == gini_org
        assert sorted(gini_ans) == gini_ans

    @mark.parametrize("lbd", [-1.0, 1.5])
    def test_lbd_bound(
        self,
        relevance_scores: np.ndarray,
        similarity_scores: np.ndarray,
        lbd: float,
    ) -> None:
        with raises(AssertionError):
            mmr = rs.MaximalMarginalRelevance(lbd)

    @mark.parametrize("scale", [0, -2])
    def test_k_bound(
        self, relevance_scores: np.ndarray, similarity_scores: np.ndarray, scale: int
    ) -> None:
        with raises(AssertionError):
            mmr = rs.MaximalMarginalRelevance(0.5)
            ret = mmr.rerank(
                relevance_scores, scale, similarity_scores=similarity_scores
            )
