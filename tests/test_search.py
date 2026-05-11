from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.search import (
    EmptySearchIndexError,
    InMemoryKeywordSearcher,
    InMemorySemanticSearcher,
    MissingEmbeddingsError,
    NoSearchIndexFound,
    TextSearchArgs,
    TopicSearchArgs,
    get_searcher,
    get_searcher_label,
    get_topic_searcher,
    run_search_by_text,
    run_search_by_topic,
)

SAMPLE_POSTS = [
    {
        "post_id": "1",
        "post_text": "Python is great for data science",
        "doc_embedding": [1.0, 0.0, 0.0],  # sample test embedding with dim 3
    },
    {
        "post_id": "2",
        "post_text": "I love machine learning with Python",
        "doc_embedding": [0.0, 1.0, 0.0],
    },
    {"post_id": "3", "post_text": "Rust is blazing fast", "doc_embedding": [0.0, 0.0, 1.0]},
]


@pytest.fixture
def keyword_searcher():
    return InMemoryKeywordSearcher(SAMPLE_POSTS)


@pytest.fixture
def semantic_searcher():
    with patch("src.search.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(
            [1.0, 0.0, 0.0]
        )  # sample test embedding with dim 3
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

    def test_missing_embeddings_error_raised_when_doc_embedding_missing(self):
        posts_without_embedding = [
            {"post_id": "1", "post_text": "Some text", "doc_embedding": None},
        ]
        with (
            patch("src.search.SentenceTransformer"),
            pytest.raises(
                MissingEmbeddingsError, match="One or more document embeddings are not stored"
            ),
        ):
            InMemorySemanticSearcher(posts_without_embedding)

    def test_missing_embeddings_error_raised_when_embedding_empty_list(self):
        posts_without_embedding = [
            {"post_id": "1", "post_text": "Some text", "doc_embedding": []},
        ]
        with patch("src.search.SentenceTransformer"), pytest.raises(MissingEmbeddingsError):
            InMemorySemanticSearcher(posts_without_embedding)

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


class TestSemanticSearcher:
    """Test SemanticSearcher (Elasticsearch-backed vector search)."""

    @pytest.fixture
    def semantic_searcher_es(self):
        """Create a SemanticSearcher with mocked Elasticsearch."""
        with (
            patch("src.search.Elasticsearch") as mock_es_cls,
            patch("src.search.SentenceTransformer") as mock_st_cls,
        ):
            mock_es = MagicMock()
            mock_es_cls.return_value = mock_es

            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([1.0, 0.0, 0.0])
            mock_st_cls.return_value = mock_model

            from src.search import SemanticSearcher

            searcher = SemanticSearcher(SAMPLE_POSTS, index_name="test_index")
            searcher.client = mock_es
            yield searcher, mock_es, mock_model

    def test_convert_keywords_to_embedding(self, semantic_searcher_es):
        searcher, _, mock_model = semantic_searcher_es
        keywords = ["python", "machine learning"]
        embedding = searcher.convert_keywords_to_embedding(keywords)
        assert isinstance(embedding, np.ndarray)
        mock_model.encode.assert_called_with("python machine learning", convert_to_numpy=True)

    def test_search_similar_documents_returns_sorted_results(self, semantic_searcher_es):
        searcher, mock_es, _ = semantic_searcher_es
        mock_es.search.return_value = {
            "hits": {
                "hits": [
                    {"_score": 0.9, "_source": {"post_id": "1", "post_text": "test"}},
                    {"_score": 0.7, "_source": {"post_id": "2", "post_text": "test2"}},
                ]
            }
        }

        query_embedding = np.array([1.0, 0.0, 0.0])
        results = searcher.search_similar_documents(query_embedding)

        assert len(results) == 2
        assert results[0]["score"] == 0.9
        assert results[1]["score"] == 0.7

    def test_search_respects_top_k(self, semantic_searcher_es):
        searcher, mock_es, _ = semantic_searcher_es
        mock_es.search.return_value = {
            "hits": {
                "hits": [
                    {"_score": 0.9, "_source": {"post_id": "1", "post_text": "test"}},
                    {"_score": 0.7, "_source": {"post_id": "2", "post_text": "test2"}},
                    {"_score": 0.5, "_source": {"post_id": "3", "post_text": "test3"}},
                ]
            }
        }

        query_embedding = np.array([1.0, 0.0, 0.0])
        searcher.search_similar_documents(query_embedding, top_k=2)

        call_args = mock_es.search.call_args
        assert call_args[1]["body"]["size"] == 2

    def test_search_accepts_keywords(self, semantic_searcher_es):
        searcher, mock_es, _ = semantic_searcher_es
        mock_es.search.return_value = {"hits": {"hits": []}}

        results = searcher.search(["python", "data"])
        assert isinstance(results, list)
        mock_es.search.assert_called_once()

    def test_search_accepts_ndarray(self, semantic_searcher_es):
        searcher, mock_es, _ = semantic_searcher_es
        mock_es.search.return_value = {"hits": {"hits": []}}

        embedding = np.array([1.0, 0.0, 0.0])
        results = searcher.search(embedding)
        assert isinstance(results, list)
        mock_es.search.assert_called_once()

    def test_search_raises_on_invalid_input(self, semantic_searcher_es):
        searcher, _, _ = semantic_searcher_es
        with pytest.raises(ValueError, match="Input must be a list of keywords"):
            searcher.search("invalid_string_input")

    def test_connection_error_raised_when_es_unavailable(self, semantic_searcher_es):
        from elastic_transport import ConnectionError as ESConnectionError

        searcher, mock_es, _ = semantic_searcher_es
        mock_es.search.side_effect = ESConnectionError("Connection failed")

        query_embedding = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ConnectionError, match="Check that the docker container is running"):
            searcher.search_similar_documents(query_embedding)

    def test_empty_search_index_error_raised_when_index_empty(self):
        """Test that EmptySearchIndexError is raised when index exists but has no documents."""
        from elasticsearch import BadRequestError

        with patch("src.search.Elasticsearch") as mock_es_cls:
            mock_es = MagicMock()
            mock_es.search.side_effect = BadRequestError(400, "body", "error_msg")
            mock_es.count.return_value = {"count": 0}
            mock_es_cls.return_value = mock_es

            from src.search import SemanticSearcher

            searcher = SemanticSearcher([], index_name="empty_index")
            query_embedding = np.array([1.0, 0.0, 0.0])

            with pytest.raises(EmptySearchIndexError, match="Index 'empty_index' is empty"):
                searcher.search_similar_documents(query_embedding)

    def test_no_search_index_found_raised_when_index_not_found(self):
        """Test that NoSearchIndexFound is raised when the index does not exist."""
        from elasticsearch import NotFoundError

        with patch("src.search.Elasticsearch") as mock_es_cls:
            mock_es = MagicMock()
            mock_es.count.side_effect = NotFoundError(
                404, "index_not_found_exception", "no such index"
            )
            mock_es_cls.return_value = mock_es

            from src.search import SemanticSearcher

            searcher = SemanticSearcher([], index_name="missing_index")
            query_embedding = np.array([1.0, 0.0, 0.0])

            with pytest.raises(NoSearchIndexFound, match="Index 'missing_index' does not exist"):
                searcher.search_similar_documents(query_embedding)

    def test_bad_request_error_reraised_when_index_has_documents(self, semantic_searcher_es):
        """Test that BadRequestError is re-raised if index has documents, but
        mapping is incorrect (e.g. due to schema mismatch)."""
        from elasticsearch import BadRequestError

        searcher, mock_es, _ = semantic_searcher_es
        mock_es.search.side_effect = BadRequestError(400, "body", "error_msg")
        mock_es.count.return_value = {"count": 5}

        query_embedding = np.array([1.0, 0.0, 0.0])
        with pytest.raises(BadRequestError):
            searcher.search_similar_documents(query_embedding)


class TestRunKeywordSearch:
    def test_uses_keyword_searcher(self, keyword_searcher):
        results = run_search_by_text(TextSearchArgs(query="Python", searcher=keyword_searcher))
        assert len(results) == 2

    def test_uses_embedding_when_searcher_has_model(self, semantic_searcher):
        results = run_search_by_text(TextSearchArgs(query="Python", searcher=semantic_searcher))
        assert isinstance(results, list)


class TestRunSemanticSearch:
    def test_returns_results(self, semantic_searcher):
        embedding = np.array([1.0, 0.0, 0.0])
        results = run_search_by_topic(
            TopicSearchArgs(embedding=embedding, searcher=semantic_searcher)
        )
        assert len(results) > 0

    def test_raises_when_searcher_lacks_embedding_model(self, keyword_searcher):
        embedding = np.array([1.0, 0.0, 0.0])
        with pytest.raises(TypeError):
            run_search_by_topic(TopicSearchArgs(embedding=embedding, searcher=keyword_searcher))
