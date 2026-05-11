from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from src.ai_labeler import build_llm_representation
from src.config import OLLAMA_MODEL, OLLAMA_URL, OPENAI_MODEL


class TestBuildLlmRepresentation:
    @patch("src.ai_labeler.BertTopicOpenAI")
    @patch("httpx.get")
    @patch("src.ai_labeler.OpenAI")
    def test_prefers_ollama_when_available(
        self, mock_openai_client, mock_httpx_get, mock_bt_openai, monkeypatch
    ) -> None:
        # Mock the /api/tags response to include the configured model
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": OLLAMA_MODEL}]}
        mock_response.content = b"{}"
        mock_httpx_get.return_value = mock_response

        mock_bt_openai.return_value = MagicMock(name="ollama_model")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        result = build_llm_representation(prompt="label this topic")

        assert result is mock_bt_openai.return_value
        mock_httpx_get.assert_called_once_with(OLLAMA_URL.replace("/v1", "/api/tags"), timeout=2)
        mock_openai_client.assert_called_once_with(base_url=OLLAMA_URL, api_key="ollama")
        assert mock_bt_openai.call_args.kwargs["model"] == OLLAMA_MODEL

    @patch("src.ai_labeler.BertTopicOpenAI")
    @patch("httpx.get")
    @patch("src.ai_labeler.OpenAI")
    def test_falls_back_to_openai_api_key(
        self, mock_openai_client, mock_httpx_get, mock_bt_openai, monkeypatch
    ) -> None:
        mock_httpx_get.side_effect = Exception("ollama down")
        mock_bt_openai.return_value = MagicMock(name="openai_model")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-live")
        monkeypatch.setenv("OPENAI_ORGANIZATION", "org-1")
        monkeypatch.setenv("OPENAI_PROJECT", "proj-1")

        result = build_llm_representation(prompt="label this topic")

        assert result is mock_bt_openai.return_value
        mock_openai_client.assert_called_once_with(
            api_key="sk-live", organization="org-1", project="proj-1"
        )
        assert mock_bt_openai.call_args.kwargs["model"] == OPENAI_MODEL

    @patch("httpx.get")
    def test_returns_none_when_no_backend_available(self, mock_httpx_get, monkeypatch) -> None:
        mock_httpx_get.side_effect = Exception("ollama down")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        result = build_llm_representation(prompt="label this topic")

        assert result is None

    @patch("httpx.get")
    def test_warns_when_ollama_model_missing(self, mock_httpx_get, caplog) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"{}"
        mock_response.json.return_value = {"models": [{"name": "some-other-model"}]}
        mock_httpx_get.return_value = mock_response

        with caplog.at_level(logging.WARNING, logger="src.ai_labeler"):
            _ = build_llm_representation(prompt="label this topic")

        assert any(
            "configured model" in record.message and "not available" in record.message
            for record in caplog.records
        )

    @patch("httpx.get")
    def test_warns_when_ollama_tags_non_200(self, mock_httpx_get, caplog) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.content = b""
        mock_httpx_get.return_value = mock_response

        with caplog.at_level(logging.WARNING, logger="src.ai_labeler"):
            _ = build_llm_representation(prompt="label this topic")

        assert any("/api/tags returned HTTP 503" in record.message for record in caplog.records)

    @patch("httpx.get")
    def test_warns_when_ollama_tags_query_fails(self, mock_httpx_get, caplog) -> None:
        mock_httpx_get.side_effect = Exception("network unreachable")

        with caplog.at_level(logging.WARNING, logger="src.ai_labeler"):
            _ = build_llm_representation(prompt="label this topic")

        assert any(
            "Could not query Ollama model tags" in record.message for record in caplog.records
        )

    @patch("httpx.get")
    def test_warns_when_no_openai_key_and_no_llm(self, mock_httpx_get, monkeypatch, caplog) -> None:
        mock_httpx_get.side_effect = Exception("ollama down")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with caplog.at_level(logging.WARNING, logger="src.ai_labeler"):
            result = build_llm_representation(prompt="label this topic")

        assert result is None
        assert any("No OpenAI API key found" in record.message for record in caplog.records)
        assert any("No LLM available" in record.message for record in caplog.records)
