from typing import List

import numpy as np
import pandas as pd
from pytest import fixture, mark, raises

import rsdiv as rs


class TestSlidingSpectrumDecomposition:
    @fixture
    def candidate_scale(self) -> int:
        return 200

    @fixture
    def dataset(self, candidate_scale: int) -> pd.DataFrame:
        loader = rs.MovieLens1MDownLoader()
        dataframe = loader.read_items()
        return dataframe.head(candidate_scale).copy()

    @fixture
    def embedder(self) -> rs.FastTextEmbedder:
        return rs.FastTextEmbedder()

    @fixture
    def relevance_scores(self, candidate_scale: int) -> np.ndarray:
        rng = np.random.default_rng(42)
        scores = -np.sort(-rng.random(candidate_scale))
        return scores

    @fixture
    def embeddings(
        self, dataset: pd.DataFrame, embedder: rs.FastTextEmbedder
    ) -> np.ndarray:
        dataframe = dataset
        dataframe["embedding"] = dataframe["genres"].apply(embedder.embedding_list)
        embedding: np.ndarray = np.asarray(list(dataframe["embedding"]))

        return embedding

    # Note: these tests calculate diversity scores of a recommendation by directly using movies' categorical genres, but
    # the diversity enhancement algorithms use genres' embeddings from fasttext to rerank. As a result, the scores might
    # decrease when gamma or scale are small. The tests here choose scale and gamma large enough to make SSD pass.
    @mark.parametrize("scale", [50, 70, 100])
    @mark.parametrize("gamma", np.linspace(20, 60, 20))
    def test_rerank_scale(
        self,
        dataset: pd.DataFrame,
        relevance_scores: np.ndarray,
        embeddings: np.ndarray,
        scale: int,
        gamma: float,
    ) -> None:
        genres = dataset["genres"]
        ssd = rs.SlidingSpectrumDecomposition(gamma)
        metrics = rs.DiversityMetrics()
        selected_ind = ssd.rerank(relevance_scores, scale, embeddings=embeddings)
        gini_org = metrics.gini_coefficient(genres[:scale])
        gini_ssd = metrics.gini_coefficient([genres[index] for index in selected_ind])
        assert selected_ind[0] == 0
        assert len(selected_ind) == scale
        assert gini_org >= gini_ssd

    @mark.parametrize("scale", [100, 120, 150, 170])
    def test_rerank_gamma(
        self,
        dataset: pd.DataFrame,
        relevance_scores: np.ndarray,
        embeddings: np.ndarray,
        scale: int,
        gamma: List[float] = np.linspace(0, 20, 20),
    ) -> None:
        genres = dataset["genres"]
        metrics = rs.DiversityMetrics()
        gini_org = metrics.gini_coefficient(genres[:scale])
        gini_ans = []
        for lbd in gamma:
            ssd = rs.SlidingSpectrumDecomposition(lbd)
            selected_ind = ssd.rerank(relevance_scores, scale, embeddings=embeddings)
            gini_ssd = metrics.gini_coefficient(
                [genres[index] for index in selected_ind]
            )
            gini_ans.append(gini_ssd)
        assert selected_ind[0] == 0
        assert gini_ans[0] == gini_org
        assert sorted(gini_ans, reverse=True) == gini_ans

    @mark.parametrize("gamma", [-1.0, -2.0])
    def test_gamma_bound(
        self,
        relevance_scores: np.ndarray,
        embeddings: np.ndarray,
        gamma: float,
    ) -> None:
        with raises(AssertionError):
            ssd = rs.SlidingSpectrumDecomposition(gamma)

    @mark.parametrize("scale", [0, -2])
    def test_k_bound(
        self, relevance_scores: np.ndarray, embeddings: np.ndarray, scale: int
    ) -> None:
        with raises(AssertionError):
            ssd = rs.SlidingSpectrumDecomposition(0.5)
            ret = ssd.rerank(relevance_scores, scale, embeddings=embeddings)
