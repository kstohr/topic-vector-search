from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.ai_labeler import build_llm_representation
from src.config import OLLAMA_MODEL, OLLAMA_URL, OPENAI_MODEL


class TestBuildLlmRepresentation:
    @patch("src.ai_labeler.BertTopicOpenAI")
    @patch("httpx.get")
    @patch("openai.OpenAI")
    def test_prefers_ollama_when_available(
        self, mock_openai_client, mock_httpx_get, mock_bt_openai, monkeypatch
    ) -> None:
        mock_httpx_get.return_value.status_code = 200
        mock_bt_openai.return_value = MagicMock(name="ollama_model")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        result = build_llm_representation(prompt="label this topic")

        assert result is mock_bt_openai.return_value
        mock_httpx_get.assert_called_once_with(OLLAMA_URL.replace("/v1", "/api/tags"), timeout=2)
        mock_openai_client.assert_called_once_with(base_url=OLLAMA_URL, api_key="ollama")
        assert mock_bt_openai.call_args.kwargs["model"] == OLLAMA_MODEL

    @patch("src.ai_labeler.BertTopicOpenAI")
    @patch("httpx.get")
    @patch("openai.OpenAI")
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
    @patch("openai.OpenAI")
    def test_returns_none_when_no_backend_available(
        self, mock_openai_client, mock_httpx_get, monkeypatch
    ) -> None:
        mock_httpx_get.side_effect = Exception("ollama down")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        result = build_llm_representation(prompt="label this topic")

        assert result is None
        mock_openai_client.assert_not_called()
