"""Elasticsearch index definition and lifecycle helpers for post_docs."""

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


def get_client() -> Elasticsearch:
    """Return a new Elasticsearch client using the configured URL."""
    return Elasticsearch(ELASTICSEARCH_URL)


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
