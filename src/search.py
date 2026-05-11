"""
====================
SEARCH
====================

Search backends: keyword (BM25/in-memory) and semantic
(Elasticsearch/in-memory).
"""

import logging
from typing import Any

import numpy as np
from elasticsearch import Elasticsearch
from pydantic import BaseModel, ConfigDict
from sentence_transformers import SentenceTransformer

from src.config import ELASTICSEARCH_URL, EMBEDDING_MODEL_NAME
from src.data_models import PostDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TOP_K_DEFAULT = 20


class NoSearchIndexFound(Exception):
    """Raised when the Elasticsearch index does not exist."""


class EmptySearchIndexError(Exception):
    """Raised when the Elasticsearch index exists but contains no documents."""


class MissingEmbeddingsError(Exception):
    """Raised when document embeddings are required but not found in the input
    data."""


class KeywordSearcher:
    """Keyword search (lexical/BM25) via Elasticsearch."""

    def __init__(self, _posts: list[dict], index_name: str = "post_docs"):
        """Initialise Elasticsearch client. _posts unused (keep for compatibility)."""
        self.index_name = index_name
        self.client = Elasticsearch(ELASTICSEARCH_URL)

    def search_similar_documents(
        self, query: str, top_k: int = TOP_K_DEFAULT, filters: list[dict] | None = None
    ) -> list[dict]:
        """Run a BM25 match query against post_text and return ranked results."""
        from elasticsearch import NotFoundError

        body = {
            "size": top_k,
            "query": {"match": {"post_text": query}},
        }
        try:
            resp = self.client.search(index=self.index_name, body=body)
        except NotFoundError as error:
            raise NoSearchIndexFound(
                f"Index '{self.index_name}' does not exist — run src.preprocess to index posts"
            ) from error
        except Exception as error:
            raise ConnectionError("Check that the docker container is running") from error
        results = [
            {
                "score": hit["_score"],
                **{k: v for k, v in hit["_source"].items() if k != "doc_embedding"},
            }
            for hit in resp["hits"]["hits"]
        ]
        return sorted(results, key=lambda result: result.get("score", 0.0), reverse=True)

    def search(self, input_data: list[str] | str, top_k: int = TOP_K_DEFAULT) -> list[dict]:
        """Accept a string or list of strings and pass to search_similar_documents."""
        query = " ".join(input_data) if isinstance(input_data, list) else str(input_data)
        return self.search_similar_documents(query, top_k)


class InMemoryKeywordSearcher:
    """Keyword search (lexical substring match) over post text — no Elasticsearch required."""

    def __init__(self, posts: list[dict]):
        """Store posts for in-memory search."""
        self.posts = posts

    def search_similar_documents(
        self, query: str, top_k: int = TOP_K_DEFAULT, filters: list[dict] | None = None
    ) -> list[dict]:
        """Count query occurrences in each post and return the top_k matches."""
        query_lower = query.lower()
        results = [
            {
                "score": post.get("post_text", "").lower().count(query_lower),
                **{key: val for key, val in post.items() if key != "doc_embedding"},
            }
            for post in self.posts
            if query_lower in post.get("post_text", "").lower()
        ]
        ranked = sorted(results, key=lambda result: result.get("score", 0.0), reverse=True)
        return ranked[:top_k]

    def search(self, input_data: list[str] | str, top_k: int = TOP_K_DEFAULT) -> list[dict]:
        """Accept a string or list of strings and delegate to search_similar_documents."""
        query = " ".join(input_data) if isinstance(input_data, list) else str(input_data)
        return self.search_similar_documents(query, top_k)


class SemanticSearcher:
    """
    Semantic search via Elasticsearch vector search.

    Reference:
    https://www.elastic.co/docs/solutions/search/vector/knn#knn-semantic-search
    https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-script-score-query.html
    """

    def __init__(
        self,
        _posts: list[dict],
        index_name: str = "post_docs",
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ):
        """Initialise embedding model and Elasticsearch client. _posts unused (factory compat)."""
        self.index_name = index_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.client = Elasticsearch(ELASTICSEARCH_URL)

    def convert_keywords_to_embedding(self, keywords: list[str]) -> np.ndarray:
        """Convert a list of keywords to an embedding using SentenceTransformers."""
        logger.info("Converting keywords to embeddings.")
        text = " ".join(keywords)  # Combine keywords into a single string
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding

    def search_similar_documents(
        self,
        embedding: np.ndarray,
        top_k: int = TOP_K_DEFAULT,
        filters: list[dict] | None = None,
    ) -> list[dict]:
        """Vector search in Elasticsearch.

        - If top_k is set: return top_k docs ranked by cosine similarity.
        - If top_k is None: return all matching docs ranked by cosine similarity.
        """
        from elasticsearch import NotFoundError

        # Check if index has any documents at all
        try:
            total_count = self.client.count(index=self.index_name).get("count", 0)
        except NotFoundError as e:
            raise NoSearchIndexFound(
                f"Index '{self.index_name}' does not exist — run src.preprocess to index posts"
            ) from e

        if total_count == 0:
            raise EmptySearchIndexError(
                f"Index '{self.index_name}' is empty — "
                f"run src.preprocess to generate embeddings and index posts"
            )

        # Check if doc_embedding field exists in the index before executing script_score
        doc_embedding_count = self.client.count(
            index=self.index_name, body={"query": {"exists": {"field": "doc_embedding"}}}
        ).get("count", 0)

        if doc_embedding_count == 0:
            raise MissingEmbeddingsError(
                "Semantic Search is not available. "
                "One or more document embeddings are not stored. "
                "Have you completed the generate_embeddings coding exercise and run preprocess.py?"
            )

        # ES requires size to be an integer; None means "return all up to the ES max"
        size = top_k if top_k is not None else 10_000

        filter_query = {"bool": {"filter": filters}} if filters else {"match_all": {}}
        body = {
            "size": size,
            "query": {
                "script_score": {
                    "query": filter_query,
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'doc_embedding') + 1.0",
                        "params": {"query_vector": embedding.tolist()},
                    },
                }
            },
        }

        from elastic_transport import ConnectionError as ESConnectionError
        from elasticsearch import BadRequestError, NotFoundError

        try:
            response = self.client.search(index=self.index_name, body=body)
        except NotFoundError as e:
            raise NoSearchIndexFound(
                f"Index '{self.index_name}' does not exist — run src.preprocess to index posts"
            ) from e
        except ESConnectionError as e:
            raise ConnectionError("Check that the docker container is running") from e
        except BadRequestError:
            raise
        hits = response["hits"]["hits"]
        results = [{"score": hit["_score"], **hit["_source"]} for hit in hits]
        results.sort(key=lambda result: result.get("score", 0.0), reverse=True)
        logger.info(f"Found {len(results)} similar documents (top_k={top_k}, size={size}).")
        return results

    def search(self, input_data: list[str] | np.ndarray, top_k: int = TOP_K_DEFAULT) -> list[dict]:
        """Accept keywords or a raw embedding and delegate to search_similar_documents."""
        # Determine if keywords are provided or if an embedding is directly provided
        if isinstance(input_data, list):
            logger.info("Keywords provided, converting to embedding.")
            embedding = self.convert_keywords_to_embedding(input_data)
        elif isinstance(input_data, np.ndarray):
            logger.info("Embedding provided directly.")
            embedding = input_data
        else:
            raise ValueError("Input must be a list of keywords or an np.ndarray embedding.")

        # Perform the search using the embedding
        return self.search_similar_documents(embedding, top_k)


class InMemorySemanticSearcher:
    """Drop-in replacement for Searcher that works without Elasticsearch.
    Uses cosine similarity over a numpy matrix of post embeddings.
    """

    def __init__(self, posts: list[dict], embedding_model_name: str = EMBEDDING_MODEL_NAME):
        """Pre-normalise post embeddings into a matrix for fast cosine scoring."""
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.posts = posts
        if not self.posts[0].get("doc_embedding"):
            raise MissingEmbeddingsError(
                "Semantic Search is not available. "
                "One or more document embeddings are not stored. "
                "Have you completed the generate_embeddings coding exercise and run preprocess.py?"
            )
        embeddings = np.array([p["doc_embedding"] for p in posts], dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings_norm = embeddings / np.where(norms == 0, 1, norms)

    def convert_keywords_to_embedding(self, keywords: list[str]) -> np.ndarray:
        """Join keywords into a single string and encode to an embedding."""
        text = " ".join(keywords)
        return self.embedding_model.encode(text, convert_to_numpy=True)

    def search_similar_documents(
        self,
        embedding: np.ndarray,
        top_k: int = TOP_K_DEFAULT,
        filters: list[dict] | None = None,
    ) -> list[dict]:
        """Score all posts by cosine similarity and return the top_k results."""
        query_embedding = embedding / (np.linalg.norm(embedding) or 1)
        scores = self.embeddings_norm @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_indices:
            result = {key: val for key, val in self.posts[i].items() if key != "doc_embedding"}
            result["score"] = float(scores[i])
            results.append(result)
        results.sort(key=lambda result: result.get("score", 0.0), reverse=True)
        return results

    def search(self, input_data: list[str] | np.ndarray, top_k: int = TOP_K_DEFAULT) -> list[dict]:
        """Accept keywords or a raw embedding and delegate to search_similar_documents."""
        if isinstance(input_data, list):
            embedding = self.convert_keywords_to_embedding(input_data)
        elif isinstance(input_data, np.ndarray):
            embedding = input_data
        else:
            raise ValueError("Input must be a list of keywords or an np.ndarray embedding.")
        return self.search_similar_documents(embedding, top_k)


_SEARCHER_LABELS = {
    "InMemoryKeywordSearcher": "Keyword · in-memory",
    "KeywordSearcher": "Keyword (BM25) · Elasticsearch",
    "InMemorySemanticSearcher": "Semantic · in-memory",
    "SemanticSearcher": "Semantic · Elasticsearch",
}


def get_searcher_label(searcher) -> str:
    """Return a human-readable label for the given searcher instance."""
    return _SEARCHER_LABELS.get(type(searcher).__name__, type(searcher).__name__)


_SEARCHER_CLASSES = {
    "InMemoryKeywordSearcher": InMemoryKeywordSearcher,
    "KeywordSearcher": KeywordSearcher,
    "InMemorySemanticSearcher": InMemorySemanticSearcher,
    "SemanticSearcher": SemanticSearcher,
}

TEXT_SEARCH_ENGINES = list(_SEARCHER_CLASSES.keys())
TOPIC_SEARCH_ENGINES = ["InMemorySemanticSearcher", "SemanticSearcher"]


def get_searcher(
    posts: list[dict],
    engine: str = "InMemorySemanticSearcher",
) -> KeywordSearcher | InMemoryKeywordSearcher | InMemorySemanticSearcher | SemanticSearcher:
    """Return the search engine for the search bar. engine must be a key in TEXT_SEARCH_ENGINES."""
    if engine not in _SEARCHER_CLASSES:
        raise ValueError(f"Unknown engine '{engine}'. Choose from: {TEXT_SEARCH_ENGINES}")
    return _SEARCHER_CLASSES[engine](posts)


def get_topic_searcher(
    posts: list[dict],
    engine: str = "InMemorySemanticSearcher",
) -> InMemorySemanticSearcher | SemanticSearcher:
    """Return search engine for topic embedding search. engine must be in TOPIC_SEARCH_ENGINES."""
    if engine not in TOPIC_SEARCH_ENGINES:
        raise ValueError(f"Unknown topic engine '{engine}'. Choose from: {TOPIC_SEARCH_ENGINES}")
    return _SEARCHER_CLASSES[engine](posts)


class TextSearchArgs(BaseModel):
    """Input arguments for run_search_by_text."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str
    searcher: Any
    top_k: int = TOP_K_DEFAULT


class TopicSearchArgs(BaseModel):
    """Input arguments for run_search_by_topic."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedding: np.ndarray
    searcher: Any
    top_k: int = TOP_K_DEFAULT


def run_search_by_text(args: TextSearchArgs) -> list[dict]:
    """Powers the search bar in the demo app. Dispatches to keyword or semantic search."""
    if hasattr(args.searcher, "embedding_model"):
        embedding = args.searcher.embedding_model.encode(args.query, convert_to_numpy=True)
        return args.searcher.search_similar_documents(embedding, top_k=args.top_k)
    return args.searcher.search_similar_documents(args.query, top_k=args.top_k)


def run_search_by_topic(args: TopicSearchArgs) -> list[dict]:
    """Powers topic embedding search. Requires a semantic searcher."""
    if not hasattr(args.searcher, "embedding_model"):
        raise TypeError(
            f"{type(args.searcher).__name__} does not support embedding-based search. "
            "Switch get_searcher() to return InMemorySemanticSearcher or SemanticSearcher."
        )
    return args.searcher.search_similar_documents(args.embedding, top_k=args.top_k)


# Example usage
if __name__ == "__main__":
    searcher = SemanticSearcher(index_name="post_docs")

    # Example 1: Search using keywords
    keywords = ["cat", "meow", "purr"]
    results = searcher.search(keywords, top_k=5)
    if results:
        results_text = [PostDocument(**result["source"]).post_text for result in results]
    print("Search results using keywords:", results_text)

    # Example 2: Search using a direct embedding
    example_embeddings = SentenceTransformer(EMBEDDING_MODEL_NAME).encode(
        ["cat", "meow", "purr"], convert_to_numpy=True
    )
    example_embedding = np.mean(
        example_embeddings, axis=0
    )  # convert to a single "document" embedding
    results = searcher.search(example_embedding, top_k=TOP_K_DEFAULT)
    if results:
        results_text = [PostDocument(**result["source"]).post_text for result in results]
        print("Search results using embedding:", results_text)
    else:
        print("No results found.")
