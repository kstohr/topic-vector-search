from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
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


class TestTrainTopicModel:
    def _make_model(self, doc_index: dict) -> TopicModeler:
        model = TopicModeler.__new__(TopicModeler)
        model.output_path = Path("output")
        model.doc_index = doc_index
        model.embedding_model = MagicMock()
        return model

    @patch("src.topic_model.BERTopic")
    @patch("src.topic_model.build_llm_representation", return_value=None)
    @patch("src.topic_model.extract_embedding_text", return_value="some text")
    def test_warns_when_no_topics_found(
        self, _mock_text, _mock_llm, mock_bertopic_cls, caplog: pytest.LogCaptureFixture
    ) -> None:
        """fit_transform assigns all docs to outlier topic (-1) -> warning logged, no exception."""
        fake_doc = MagicMock()
        fake_doc.doc_embedding = [0.1, 0.2, 0.3]

        topic_info = pd.DataFrame({"Topic": [-1]})
        instance = MagicMock()
        instance.fit_transform.return_value = ([-1], np.array([[1.0]]))
        instance.get_topic_info.return_value = topic_info
        mock_bertopic_cls.return_value = instance

        model = self._make_model({"p1": fake_doc})

        with caplog.at_level(logging.WARNING, logger="src.topic_model"):
            topics, _ = model.train_topic_model()

        assert topics == [-1]
        assert any("No valid topics found" in r.message for r in caplog.records)


class TestOutlierOnly:
    def test_save_model_warns_and_skips_when_no_valid_topics(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        model = TopicModeler.__new__(TopicModeler)
        model.output_path = Path("output")
        model.topic_model = MagicMock()
        model.embedding_model = MagicMock()
        model.n_topics = 0

        with caplog.at_level(logging.WARNING, logger="src.topic_model"):
            model._save_model()

        model.topic_model.save.assert_not_called()
        assert any(
            "Storing model will fail. Skipping model save." in r.message for r in caplog.records
        )

    def test_generate_visualizations_warns_and_skips_when_no_valid_topics(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        model = TopicModeler.__new__(TopicModeler)
        model.output_path = Path("output")
        model.topic_model = MagicMock()
        model.n_topics = 0

        with caplog.at_level(logging.WARNING, logger="src.topic_model"):
            model.generate_visualizations()

        model.topic_model.visualize_topics.assert_not_called()
        model.topic_model.visualize_barchart.assert_not_called()
        model.topic_model.visualize_heatmap.assert_not_called()
        model.topic_model.visualize_hierarchy.assert_not_called()
        model.topic_model.visualize_term_rank.assert_not_called()
        assert any(
            "No topic model or valid topics available for visualization" in r.message
            for r in caplog.records
        )
