"""
====================
PRELOAD RAW SAMPLE POSTS INTO ELASTICSEARCH
====================
Workshop pre-loading: index raw posts into Elasticsearch for keyword search.

"""

import json
import logging
from pathlib import Path

from elasticsearch import Elasticsearch, helpers

from src.config import ELASTICSEARCH_URL, REPO
from src.data_models import PostDocument
from src.es_index import INDEX_NAME, create_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def load_postdocs(path: Path) -> list[PostDocument]:
    """Load and validate post documents from a JSON file."""
    with open(path) as f:
        raw = json.load(f)
    postdocs = [PostDocument(**p) for p in raw]
    logger.info(f"Loaded {len(postdocs)} postdocs from {path.name}.")
    return postdocs


def index_postdocs(client: Elasticsearch, postdocs: list[PostDocument]) -> None:
    """Bulk-index postdocs into Elasticsearch."""
    actions = [
        {
            "_index": INDEX_NAME,
            "_id": postdoc.post_id,
            "_source": postdoc.model_dump(mode="json"),
        }
        for postdoc in postdocs
    ]
    success, failed = helpers.bulk(client, actions)
    logger.info(f"Indexed {success} postdocs into '{INDEX_NAME}'.")
    if failed:
        logger.error(f"Failed to index {len(failed)} postdocs.")


def run() -> None:
    """Connect to Elasticsearch, create the index, and load all posts."""
    client = Elasticsearch(ELASTICSEARCH_URL)
    try:
        client.info()
    except Exception as e:
        logger.error(f"Cannot connect to Elasticsearch at {ELASTICSEARCH_URL} — {e}")
        return

    create_index(client)
    postdocs = load_postdocs(REPO / "sample_posts.json")
    index_postdocs(client, postdocs)
    logger.info("Pre-loading complete. Keyword search is ready in the demo app.")
    logger.info("Run 'uv run python -m src.preprocess' to add embeddings for semantic search.")


if __name__ == "__main__":
    run()
