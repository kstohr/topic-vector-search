from __future__ import annotations

from pathlib import Path

import pytest

from src.topic_model import ProccessedPostsNotFoundError, TopicModeler


class TestRetrievePostDocuments:
    def test_raises_custom_error_when_processed_posts_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise_missing(*_args, **_kwargs):
            raise FileNotFoundError("processed_posts.json not found")

        monkeypatch.setattr("src.topic_model.retrieve_postdocs_from_disk", _raise_missing)

        model = TopicModeler.__new__(TopicModeler)
        model.elasticsearch_client = None
        model.output_path = Path("output")
        model.doc_index = {}

        with pytest.raises(
            ProccessedPostsNotFoundError, match="Make sure you have run preprocessing.py"
        ):
            model.retrieve_post_documents()

    def test_prefers_elasticsearch_when_client_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expected = {"p1": object()}
        monkeypatch.setattr(
            "src.topic_model.retrieve_postdocs_from_elasticsearch", lambda _client: expected
        )

        model = TopicModeler.__new__(TopicModeler)
        model.elasticsearch_client = object()
        model.output_path = Path("output")
        model.doc_index = {}

        result = model.retrieve_post_documents()

        assert result is expected
