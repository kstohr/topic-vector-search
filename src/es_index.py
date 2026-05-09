"""
====================
ELASTICSEARCH INDEX DEFINITION AND TOOLS
====================
Elasticsearch index definition for post_docs. Includes database helpers for
creating, deleting, and managing the index.
"""

import logging

from elasticsearch import Elasticsearch

from src.config import ELASTICSEARCH_URL

logger = logging.getLogger(__name__)

INDEX_NAME = "post_docs"
INDEX_BODY = {
    "settings": {"number_of_shards": 1},
    "mappings": {
        "properties": {
            "post_id": {"type": "keyword"},
            "post_author": {"type": "keyword"},
            "created_at": {"type": "date"},
            "modified_at": {"type": "date"},
            "post_text": {"type": "text"},
            "doc_embedding": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine",  # Distance metric
            },
        }
    },
}


def get_es_client() -> Elasticsearch:
    """Return a new Elasticsearch client using the configured URL."""
    return Elasticsearch(ELASTICSEARCH_URL, request_timeout=30)


def create_index(client: Elasticsearch) -> None:
    """Create the post_docs index if it does not already exist."""
    if not client.indices.exists(index=INDEX_NAME):
        client.indices.create(index=INDEX_NAME, body=INDEX_BODY)
        logger.info(f"Index '{INDEX_NAME}' created successfully.")
    else:
        logger.info(f"Index '{INDEX_NAME}' already exists.")


def delete_index(client: Elasticsearch) -> None:
    """Delete the post_docs index if it exists."""
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
        logger.info(f"Index '{INDEX_NAME}' deleted successfully.")
    else:
        logger.info(f"Index '{INDEX_NAME}' does not exist.")


def count_documents(client: Elasticsearch) -> int:
    """Return the number of documents stored in the post_docs index."""
    if not client.indices.exists(index=INDEX_NAME):
        return 0
    result = client.count(index=INDEX_NAME)
    return result["count"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    es_client = get_es_client()
    count = count_documents(es_client)
    logger.info(f"Index contains {count} documents.")
