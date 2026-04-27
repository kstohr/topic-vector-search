from __future__ import annotations

import pytest

from src.models import PostDocument

BASE = {
    "post_id": "abc",
    "post_author": "user_1",
    "created_at": "2024-01-01T12:00:00",
    "modified_at": "2024-01-02T08:00:00",
    "post_text": "Hello world!",
    "likes": 5,
}


def make_post(**kwargs) -> PostDocument:
    return PostDocument(**{**BASE, **kwargs})


class TestPostDocumentDefaults:
    def test_creates_with_required_fields(self) -> None:
        post = make_post()
        assert post.post_id == "abc"
        assert post.likes == 5
        assert post.doc_embedding == []
        assert post.image_url is None

    def test_datetime_naive_gets_utc(self) -> None:
        post = make_post(created_at="2024-06-15T10:00:00")
        assert "+00:00" in post.created_at.isoformat()

    def test_datetime_with_tz_converts_to_utc(self) -> None:
        post = make_post(created_at="2024-06-15T10:00:00+05:00")
        assert post.created_at.utcoffset().total_seconds() == 0


class TestPreprocessText:
    def test_lowercases_text(self) -> None:
        post = make_post(post_text="HELLO WORLD")
        assert post.preprocess_text() == "hello world"

    def test_strips_punctuation(self) -> None:
        post = make_post(post_text="Hello, world!")
        result = post.preprocess_text()
        assert "," not in result
        assert "!" not in result

    def test_preserves_contractions(self) -> None:
        post = make_post(post_text="It's a beautiful day")
        assert "it's" in post.preprocess_text()

    def test_converts_emoji_to_text(self) -> None:
        post = make_post(post_text="I love cats 🐱")
        result = post.preprocess_text()
        assert "cat" in result
        assert "🐱" not in result

    def test_collapses_whitespace(self) -> None:
        post = make_post(post_text="too   many    spaces")
        assert "  " not in post.preprocess_text()

    def test_empty_text_returns_empty(self) -> None:
        post = make_post(post_text="")
        assert post.preprocess_text() == ""


class TestPreprocessSentences:
    def test_splits_on_period(self) -> None:
        post = make_post(post_text="First sentence. Second sentence.")
        sentences = post.preprocess_sentences()
        assert len(sentences) == 2

    def test_single_sentence_returns_one_item(self) -> None:
        post = make_post(post_text="Just one sentence")
        assert len(post.preprocess_sentences()) == 1
