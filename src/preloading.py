"""
Workshop pre-loading script: index posts into Elasticsearch for keyword search.

Run this before the workshop so attendees can test BM25 keyword search
without needing to run the full embedding pipeline first.

Run:
    uv run python -m src.preloading

What it does:
  1. Connects to Elasticsearch at ELASTICSEARCH_URL (default localhost:9201)
  3. Bulk-indexes all posts from sample_posts.json

After this runs, the Elasticsearch-backed KeywordSearcher and the
in-memory InMemoryKeywordSearcher both work in the demo app.
Run src/preprocess.py to add embeddings and unlock semantic search.
"""

import json
import logging
from pathlib import Path

from elasticsearch import Elasticsearch, helpers

from src.config import ELASTICSEARCH_URL, REPO
from src.index import INDEX_NAME, create_index
from src.models import PostDocument

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def load_posts(path: Path) -> list[PostDocument]:
    with open(path) as f:
        raw = json.load(f)
    posts = [PostDocument(**p) for p in raw]
    logger.info(f"Loaded {len(posts)} posts from {path.name}.")
    return posts


def index_posts(client: Elasticsearch, posts: list[PostDocument]) -> None:
    actions = [
        {
            "_index": INDEX_NAME,
            "_id": post.post_id,
            "_source": post.model_dump(mode="json"),
        }
        for post in posts
    ]
    success, failed = helpers.bulk(client, actions)
    logger.info(f"Indexed {success} posts into '{INDEX_NAME}'.")
    if failed:
        logger.error(f"Failed to index {len(failed)} posts.")


def run() -> None:
    client = Elasticsearch(ELASTICSEARCH_URL)
    try:
        client.info()
    except Exception as e:
        logger.error(f"Cannot connect to Elasticsearch at {ELASTICSEARCH_URL} — {e}")
        return

    create_index(client)
    posts = load_posts(REPO / "sample_posts.json")
    index_posts(client, posts)
    logger.info("Pre-loading complete. Keyword search is ready in the demo app.")
    logger.info("Run 'uv run python -m src.preprocess' to add embeddings for semantic search.")


if __name__ == "__main__":
    run()
