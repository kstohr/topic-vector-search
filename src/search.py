import logging
from typing import List, Union

import numpy as np
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME
from src.models import PostDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeywordSearcher:
    """Keyword search (lexical/BM25) via Elasticsearch."""

    def __init__(self, posts: List[dict], index_name: str = "post_docs"):
        self.index_name = index_name
        self.client = Elasticsearch("http://localhost:9200")

    def search_similar_documents(
        self, query: str, top_k: int = 20, filters: List[dict] = None
    ) -> List[dict]:
        body = {
            "size": top_k,
            "query": {"match": {"post_text": query}},
        }
        resp = self.client.search(index=self.index_name, body=body)
        return [
            {"score": hit["_score"], **{k: v for k, v in hit["_source"].items() if k != "doc_embedding"}}
            for hit in resp["hits"]["hits"]
        ]

    def search(self, input_data: Union[List[str], str], top_k: int = 20) -> List[dict]:
        query = " ".join(input_data) if isinstance(input_data, list) else str(input_data)
        return self.search_similar_documents(query, top_k)


class InMemoryKeywordSearcher:
    """Keyword search (lexical substring match) over post text — no Elasticsearch required."""

    def __init__(self, posts: List[dict]):
        self.posts = posts

    def search_similar_documents(
        self, query: str, top_k: int = 20, filters: List[dict] = None
    ) -> List[dict]:
        q = query.lower()
        results = [
            {"score": 1.0, **{k: v for k, v in p.items() if k != "doc_embedding"}}
            for p in self.posts
            if q in p.get("post_text", "").lower()
        ]
        return results[:top_k]

    def search(self, input_data: Union[List[str], str], top_k: int = 20) -> List[dict]:
        query = " ".join(input_data) if isinstance(input_data, list) else str(input_data)
        return self.search_similar_documents(query, top_k)


class SemanticSearcher:
    def __init__(self, posts: List[dict], index_name: str = "post_docs", embedding_model_name: str = EMBEDDING_MODEL_NAME):
        self.index_name = index_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.client = Elasticsearch("http://localhost:9200")

    def convert_keywords_to_embedding(self, keywords: List[str]) -> np.ndarray:
        """Convert a list of keywords to an embedding using SentenceTransformers."""
        logger.info("Converting keywords to embeddings.")
        text = " ".join(keywords)  # Combine keywords into a single string
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding

    def search_similar_documents(
        self,
        embedding: np.ndarray,
        top_k: int = 50,
        filters: List[dict] = None,
    ) -> List[dict]:
        """Search for similar documents in Elasticsearch using kNN."""
        logger.info("Searching for similar documents in Elasticsearch.")
        body = {
            "knn": {
                "field": "doc_embedding",
                "query_vector": embedding.tolist(),
                "k": top_k,
                "num_candidates": min(top_k * 20, 10000),
            },
            "size": top_k,
        }
        if filters:
            body["knn"]["filter"] = {"bool": {"must": filters}}
        response = self.client.search(index=self.index_name, body=body)
        hits = response["hits"]["hits"]
        results = [{"score": hit["_score"], **hit["_source"]} for hit in hits]
        logger.info(f"Found {len(results)} similar documents.")
        return results

    def search(
        self, input_data: Union[List[str], np.ndarray], top_k: int = 5
    ) -> List[dict]:
        """
        Search for similar documents using either a list of keywords or a provided embedding.
        :param input_data: A list of keywords to convert to an embedding or an embedding itself.
        :param top_k: The number of top similar documents to retrieve.
        :return: A list of similar documents from Elasticsearch.
        """
        # Determine if keywords are provided or if an embedding is directly provided
        if isinstance(input_data, list):
            logger.info("Keywords provided, converting to embedding.")
            embedding = self.convert_keywords_to_embedding(input_data)
        elif isinstance(input_data, np.ndarray):
            logger.info("Embedding provided directly.")
            embedding = input_data
        else:
            raise ValueError(
                "Input must be a list of keywords or an np.ndarray embedding."
            )

        # Perform the search using the embedding
        return self.search_similar_documents(embedding, top_k)





class InMemorySemanticSearcher:
    """Drop-in replacement for Searcher that works without Elasticsearch.
    Uses cosine similarity over a numpy matrix of post embeddings.
    """

    def __init__(self, posts: List[dict], embedding_model_name: str = EMBEDDING_MODEL_NAME):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.posts = posts
        embeddings = np.array([p["doc_embedding"] for p in posts], dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings_norm = embeddings / np.where(norms == 0, 1, norms)

    def convert_keywords_to_embedding(self, keywords: List[str]) -> np.ndarray:
        text = " ".join(keywords)
        return self.embedding_model.encode(text, convert_to_numpy=True)

    def search_similar_documents(
        self,
        embedding: np.ndarray,
        top_k: int = 20,
        filters: List[dict] = None,
    ) -> List[dict]:
        q = embedding / (np.linalg.norm(embedding) or 1)
        scores = self.embeddings_norm @ q
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_indices:
            result = {k: v for k, v in self.posts[i].items() if k != "doc_embedding"}
            result["score"] = float(scores[i])
            results.append(result)
        return results

    def search(self, input_data: Union[List[str], np.ndarray], top_k: int = 20) -> List[dict]:
        if isinstance(input_data, list):
            embedding = self.convert_keywords_to_embedding(input_data)
        elif isinstance(input_data, np.ndarray):
            embedding = input_data
        else:
            raise ValueError("Input must be a list of keywords or an np.ndarray embedding.")
        return self.search_similar_documents(embedding, top_k)


_SEARCHER_LABELS = {
    "InMemoryKeywordSearcher":  "Keyword · in-memory",
    "KeywordSearcher":          "Keyword (BM25) · Elasticsearch",
    "InMemorySemanticSearcher": "Semantic · in-memory",
    "SemanticSearcher":         "Semantic · Elasticsearch",
}


def get_searcher_label(searcher) -> str:
    return _SEARCHER_LABELS.get(type(searcher).__name__, type(searcher).__name__)


_SEARCHER_CLASSES = {
    "InMemoryKeywordSearcher":  InMemoryKeywordSearcher,
    "KeywordSearcher":          KeywordSearcher,
    "InMemorySemanticSearcher": InMemorySemanticSearcher,
    "SemanticSearcher":         SemanticSearcher,
}

TEXT_SEARCH_ENGINES = list(_SEARCHER_CLASSES.keys())
TOPIC_SEARCH_ENGINES = ["InMemorySemanticSearcher", "SemanticSearcher"]


def get_searcher(
    posts: List[dict],
    engine: str = "InMemorySemanticSearcher",
) -> KeywordSearcher | InMemoryKeywordSearcher | InMemorySemanticSearcher | SemanticSearcher:
    """Return the search engine for the search bar. engine must be a key in TEXT_SEARCH_ENGINES."""
    if engine not in _SEARCHER_CLASSES:
        raise ValueError(f"Unknown engine '{engine}'. Choose from: {TEXT_SEARCH_ENGINES}")
    return _SEARCHER_CLASSES[engine](posts)


def get_topic_searcher(
    posts: List[dict],
    engine: str = "InMemorySemanticSearcher",
) -> InMemorySemanticSearcher | SemanticSearcher:
    """Return the search engine for topic embedding search. engine must be a key in TOPIC_SEARCH_ENGINES."""
    if engine not in TOPIC_SEARCH_ENGINES:
        raise ValueError(f"Unknown topic engine '{engine}'. Choose from: {TOPIC_SEARCH_ENGINES}")
    return _SEARCHER_CLASSES[engine](posts)


def run_keyword_search(query: str, searcher, top_k: int = 20) -> List[dict]:
    if hasattr(searcher, "embedding_model"):
        embedding = searcher.embedding_model.encode(query, convert_to_numpy=True)
        return searcher.search_similar_documents(embedding, top_k=top_k)
    return searcher.search_similar_documents(query, top_k=top_k)


def run_semantic_search(embedding: np.ndarray, searcher, top_k: int = 20) -> List[dict]:
    if not hasattr(searcher, "embedding_model"):
        raise TypeError(
            f"{type(searcher).__name__} does not support embedding-based search. "
            "Switch get_searcher() to return InMemorySemanticSearcher or SemanticSearcher."
        )
    return searcher.search_similar_documents(embedding, top_k=top_k)


# Example usage
if __name__ == "__main__":
    # from source import main
    # main()  # Load, process, and store posts in Elasticsearch

    searcher = SemanticSearcher(index_name="post_docs")

    # Example 1: Search using keywords
    keywords = ["cat", "meow", "purr"]
    results = searcher.search(keywords, top_k=5)
    if results:
        results_text = [
            PostDocument(**result["source"]).post_text for result in results
        ]
    print("Search results using keywords:", results_text)

    # Example 2: Search using a direct embedding
    example_embeddings = SentenceTransformer(EMBEDDING_MODEL_NAME).encode(
        ["cat", "meow", "purr"], convert_to_numpy=True
    )
    example_embedding = np.mean(
        example_embeddings, axis=0
    )  # convert to a single "document" embedding
    results = searcher.search(example_embedding, top_k=10)
    if results:
        results_text = [
            PostDocument(**result["source"]).post_text for result in results
        ]
        print("Search results using embedding:", results_text)
    else:
        print("No results found.")
