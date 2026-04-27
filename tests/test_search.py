from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.search import (
    InMemoryKeywordSearcher,
    InMemorySemanticSearcher,
    get_searcher,
    get_searcher_label,
    get_topic_searcher,
    run_keyword_search,
    run_semantic_search,
)

SAMPLE_POSTS = [
    {"post_id": "1", "post_text": "Python is great for data science", "doc_embedding": [1.0, 0.0, 0.0]},
    {"post_id": "2", "post_text": "I love machine learning with Python", "doc_embedding": [0.0, 1.0, 0.0]},
    {"post_id": "3", "post_text": "Rust is blazing fast", "doc_embedding": [0.0, 0.0, 1.0]},
]


@pytest.fixture
def keyword_searcher():
    return InMemoryKeywordSearcher(SAMPLE_POSTS)


@pytest.fixture
def semantic_searcher():
    with patch("src.search.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([1.0, 0.0, 0.0]) # sample test embedding with dim 3
        mock_st.return_value = mock_model
        yield InMemorySemanticSearcher(SAMPLE_POSTS)


class TestInMemoryKeywordSearcher:
    def test_returns_matching_posts(self, keyword_searcher):
        results = keyword_searcher.search_similar_documents("Python")
        assert len(results) == 2

    def test_returns_empty_for_no_match(self, keyword_searcher):
        results = keyword_searcher.search_similar_documents("javascript")
        assert results == []

    def test_respects_top_k(self, keyword_searcher):
        results = keyword_searcher.search_similar_documents("Python", top_k=1)
        assert len(results) == 1

    def test_excludes_doc_embedding(self, keyword_searcher):
        results = keyword_searcher.search_similar_documents("Python")
        assert all("doc_embedding" not in r for r in results)

    def test_results_sorted_by_score_descending(self, keyword_searcher):
        results = keyword_searcher.search_similar_documents("Python")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_accepts_string(self, keyword_searcher):
        results = keyword_searcher.search("Python")
        assert len(results) == 2

    def test_search_accepts_list(self, keyword_searcher):
        results = keyword_searcher.search(["Python"])
        assert len(results) > 0

    def test_score_reflects_term_frequency(self, keyword_searcher):
        posts = [
            {"post_id": "a", "post_text": "Python Python Python", "doc_embedding": []},
            {"post_id": "b", "post_text": "Python", "doc_embedding": []},
        ]
        searcher = InMemoryKeywordSearcher(posts)
        results = searcher.search_similar_documents("Python")
        assert results[0]["post_id"] == "a"


class TestInMemorySemanticSearcher:
    def test_returns_results_sorted_by_cosine_similarity(self, semantic_searcher):
        query_embedding = np.array([1.0, 0.0, 0.0])
        results = semantic_searcher.search_similar_documents(query_embedding)
        assert results[0]["post_id"] == "1"

    def test_respects_top_k(self, semantic_searcher):
        query_embedding = np.array([1.0, 0.0, 0.0])
        results = semantic_searcher.search_similar_documents(query_embedding, top_k=1)
        assert len(results) == 1

    def test_excludes_doc_embedding(self, semantic_searcher):
        query_embedding = np.array([1.0, 0.0, 0.0])
        results = semantic_searcher.search_similar_documents(query_embedding)
        assert all("doc_embedding" not in r for r in results)

    def test_search_accepts_list_of_keywords(self, semantic_searcher):
        results = semantic_searcher.search(["python", "data"])
        assert len(results) == len(SAMPLE_POSTS)

    def test_search_accepts_ndarray(self, semantic_searcher):
        embedding = np.array([1.0, 0.0, 0.0])
        results = semantic_searcher.search(embedding)
        assert len(results) == len(SAMPLE_POSTS)

    def test_search_raises_on_invalid_input(self, semantic_searcher):
        with pytest.raises(ValueError):
            semantic_searcher.search("not a list or array")

    def test_scores_are_floats(self, semantic_searcher):
        results = semantic_searcher.search(np.array([1.0, 0.0, 0.0]))
        assert all(isinstance(r["score"], float) for r in results)


class TestGetSearcher:
    @pytest.mark.parametrize("engine", ["InMemoryKeywordSearcher", "InMemorySemanticSearcher"])
    def test_returns_correct_type_for_in_memory_engines(self, engine):
        with patch("src.search.SentenceTransformer"):
            searcher = get_searcher(SAMPLE_POSTS, engine=engine)
            assert type(searcher).__name__ == engine

    def test_raises_for_unknown_engine(self):
        with pytest.raises(ValueError, match="Unknown engine"):
            get_searcher(SAMPLE_POSTS, engine="NonExistentEngine")

    def test_default_engine_is_in_memory_semantic(self):
        with patch("src.search.SentenceTransformer"):
            searcher = get_searcher(SAMPLE_POSTS)
            assert isinstance(searcher, InMemorySemanticSearcher)


class TestGetTopicSearcher:
    def test_returns_in_memory_semantic_searcher(self):
        with patch("src.search.SentenceTransformer"):
            searcher = get_topic_searcher(SAMPLE_POSTS, engine="InMemorySemanticSearcher")
            assert isinstance(searcher, InMemorySemanticSearcher)

    def test_raises_for_keyword_engine(self):
        with pytest.raises(ValueError, match="Unknown topic engine"):
            get_topic_searcher(SAMPLE_POSTS, engine="InMemoryKeywordSearcher")


class TestGetSearcherLabel:
    def test_known_searcher_returns_label(self):
        searcher = InMemoryKeywordSearcher(SAMPLE_POSTS)
        label = get_searcher_label(searcher)
        assert label == "Keyword · in-memory"

    def test_unknown_searcher_returns_class_name(self):
        class CustomSearcher:
            pass
        label = get_searcher_label(CustomSearcher())
        assert label == "CustomSearcher"


class TestRunKeywordSearch:
    def test_uses_keyword_searcher(self, keyword_searcher):
        results = run_keyword_search("Python", keyword_searcher)
        assert len(results) == 2

    def test_uses_embedding_when_searcher_has_model(self, semantic_searcher):
        results = run_keyword_search("Python", semantic_searcher)
        assert isinstance(results, list)


class TestRunSemanticSearch:
    def test_returns_results(self, semantic_searcher):
        embedding = np.array([1.0, 0.0, 0.0])
        results = run_semantic_search(embedding, semantic_searcher)
        assert len(results) > 0

    def test_raises_when_searcher_lacks_embedding_model(self, keyword_searcher):
        embedding = np.array([1.0, 0.0, 0.0])
        with pytest.raises(TypeError):
            run_semantic_search(embedding, keyword_searcher)
