import logging
import os
from typing import List, Union

import numpy as np
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

from models import PostDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Searcher:
    def __init__(self, index_name: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.index_name = index_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.opensearch_client = OpenSearch(
            hosts=[{"host": "localhost", "port": 9200}], http_compress=True
        )

    def convert_keywords_to_embedding(self, keywords: List[str]) -> np.ndarray:
        """Convert a list of keywords to an embedding using SentenceTransformers."""
        logger.info("Converting keywords to embeddings.")
        text = " ".join(keywords)  # Combine keywords into a single string
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding

    def search_similar_documents(
        self,
        embedding: np.ndarray,
        top_k: int = 1000,  # effectively no limit
        filters: List[dict] = None,
    ) -> List[dict]:
        """Search for similar documents in OpenSearch using an embedding."""
        logger.info("Searching for similar documents in OpenSearch.")

        filters = filters or []

        query = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "knn_score",
                                    "lang": "knn",
                                    "params": {
                                        "field": "doc_embedding",
                                        "query_value": embedding.tolist(),
                                        "space_type": "cosinesimil",
                                    },
                                },
                            }
                        }
                    ],
                    "filter": filters,  # Apply filters if provided
                }
            },
            "sort": [{"_score": {"order": "desc"}}],
        }
        response = self.opensearch_client.search(index=self.index_name, body=query)
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
        :return: A list of similar documents from OpenSearch.
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


# Example usage
if __name__ == "__main__":
    from source import main

    # main()  # Load, process, and store posts in OpenSearch

    searcher = Searcher(index_name="post_docs")

    # Example 1: Search using keywords
    keywords = ["cat", "meow", "purr"]
    results = searcher.search(keywords, top_k=5)
    if results:
        results_text = [
            PostDocument(**result["source"]).post_text for result in results
        ]
    print("Search results using keywords:", results_text)

    # Example 2: Search using a direct embedding
    example_embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(
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
