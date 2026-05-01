"""
====================
RETREIVE POST DOCUMENTS
====================
Helper functions to retrieve post documents from Elasticsearch or disk.
"""

import json
import logging
from pathlib import Path

from elasticsearch import Elasticsearch

from src.data_models import PostDocument
from src.es_index import INDEX_NAME

logger = logging.getLogger(__name__)


def retrieve_postdocs_from_elasticsearch(es_client: Elasticsearch) -> dict[str, PostDocument]:
    """Scroll through Elasticsearch to retrieve all post documents."""
    doc_index: dict[str, PostDocument] = {}
    if not es_client.indices.exists(index=INDEX_NAME):
        logger.warning(
            f"Elasticsearch index '{INDEX_NAME}' does not exist. "
            "Run 'uv run python -m src.preprocess' first, or falling back to disk."
        )
        return doc_index

    logger.info("Retrieving post documents from Elasticsearch.")
    response = es_client.search(
        index=INDEX_NAME,
        scroll="2m",
        body={"size": 500, "query": {"match_all": {}}},
    )
    scroll_id = response["_scroll_id"]
    hits = response["hits"]["hits"]
    while hits:
        for hit in hits:
            postdoc = PostDocument(**hit["_source"])
            doc_index[postdoc.post_id] = postdoc
        response = es_client.scroll(scroll_id=scroll_id, scroll="2m")
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]
    es_client.clear_scroll(scroll_id=scroll_id)
    logger.info(f"Retrieved {len(doc_index)} posts from Elasticsearch.")
    return doc_index


def retrieve_postdocs_from_disk(output_path: Path) -> dict[str, PostDocument]:
    """If Elasticsearch is unavailable, load post documents from
    output/processed_posts.json."""
    doc_index: dict[str, PostDocument] = {}
    path = output_path / "processed_posts.json"
    logger.info(f"Loading posts from {path}.")
    with open(path) as f:
        raw = json.load(f)
    for post_id, data in raw.items():
        doc_index[post_id] = PostDocument(**data)
    logger.info(f"Loaded {len(doc_index)} posts from disk.")
    return doc_index
