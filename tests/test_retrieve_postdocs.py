from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.retrieve_postdocs import retrieve_postdocs_from_disk, retrieve_postdocs_from_elasticsearch

BASE_POST = {
    "post_id": "post-1",
    "post_author": "author-1",
    "created_at": "2024-01-01T00:00:00",
    "modified_at": "2024-01-01T00:00:00",
    "post_text": "hello world",
    "likes": 1,
}


def _es_hit(post_id: str, text: str) -> dict:
    return {
        "_source": {
            **BASE_POST,
            "post_id": post_id,
            "post_text": text,
            "doc_embedding": [0.1, 0.2, 0.3],
        }
    }


class TestRetrievePostdocsFromElasticsearch:
    def test_returns_empty_when_index_missing(self) -> None:
        es_client = MagicMock()
        es_client.indices.exists.return_value = False

        result = retrieve_postdocs_from_elasticsearch(es_client)

        assert result == {}
        es_client.search.assert_not_called()

    def test_scrolls_until_exhausted_and_clears_scroll(self) -> None:
        es_client = MagicMock()
        es_client.indices.exists.return_value = True
        es_client.search.return_value = {
            "_scroll_id": "scroll-1",
            "hits": {"hits": [_es_hit("post-1", "first")]},
        }
        es_client.scroll.side_effect = [
            {
                "_scroll_id": "scroll-2",
                "hits": {"hits": [_es_hit("post-2", "second")]},
            },
            {
                "_scroll_id": "scroll-3",
                "hits": {"hits": []},
            },
        ]

        result = retrieve_postdocs_from_elasticsearch(es_client)

        assert set(result.keys()) == {"post-1", "post-2"}
        assert result["post-1"].post_text == "first"
        assert result["post-2"].post_text == "second"
        es_client.clear_scroll.assert_called_once_with(scroll_id="scroll-3")


class TestRetrievePostdocsFromDisk:
    def test_loads_processed_posts_json(self, tmp_path) -> None:
        payload = {
            "post-1": {**BASE_POST, "post_id": "post-1", "doc_embedding": [0.9, 0.1]},
            "post-2": {**BASE_POST, "post_id": "post-2", "post_text": "other", "doc_embedding": []},
        }
        path = tmp_path / "processed_posts.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        result = retrieve_postdocs_from_disk(tmp_path)

        assert set(result.keys()) == {"post-1", "post-2"}
        assert result["post-1"].doc_embedding == [0.9, 0.1]
        assert result["post-2"].post_text == "other"
