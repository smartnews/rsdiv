from typing import List

import numpy as np
import pandas as pd
import pytest
from pytest import mark

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
        rng = np.random.RandomState(42)
        scores = -np.sort(-rng.rand(candidate_scale))
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

    def test_rerank_scale(
        self,
        dataset: pd.DataFrame,
        relevance_scores: np.ndarray,
        similarity_scores: np.ndarray,
        lbd: float = 0.5,
        scale_list: List[int] = [50, 70, 100],
    ) -> None:
        genres = dataset["genres"]
        mmr = rs.MaximalMarginalRelevance(lbd)
        metrics = rs.DiversityMetrics()
        for k in scale_list:
            selected_ind = mmr.rerank(
                relevance_scores, k, similarity_scores=similarity_scores
            )
            gini_org = metrics.gini_coefficient(genres[:k])
            gini_mmr = metrics.gini_coefficient(
                [genres[index] for index in selected_ind]
            )
            assert selected_ind[0] == 0
            assert len(selected_ind) == k
            assert gini_org > gini_mmr

    def test_rerank_lambda(
        self,
        dataset: pd.DataFrame,
        relevance_scores: np.ndarray,
        similarity_scores: np.ndarray,
        lbd_list: List[float] = [0, 0.3, 0.6, 1],
        scale: int = 100,
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

    def test_domain(
        self,
        relevance_scores: np.ndarray,
        similarity_scores: np.ndarray,
        lbd_list: List[float] = [1, 1.5],
        scale: int = 50,
    ) -> None:
        selected_org = list(range(scale))
        for lbd in lbd_list:
            mmr = rs.MaximalMarginalRelevance(lbd)
            selected_ind = mmr.rerank(
                relevance_scores, scale, similarity_scores=similarity_scores
            )
            assert selected_ind == selected_org
