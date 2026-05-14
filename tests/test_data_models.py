from __future__ import annotations

import pytest

from src.data_models import PostDocument

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


@pytest.mark.exercise
class TestPreprocessText:
    def test_excludes_urls(self) -> None:
        post = make_post(post_text="Check https://example.com now!")
        assert post.preprocess_text() == "Check now!"

    def test_converts_emoji_to_text(self) -> None:
        post = make_post(post_text="I love cats 🐱")
        result = post.preprocess_text()
        assert ":cat face:" in result

    def test_collapses_internal_whitespace(self) -> None:
        post = make_post(post_text="too   many    spaces")
        assert post.preprocess_text() == "too many spaces"

    def test_strips_outer_whitespace(self) -> None:
        post = make_post(post_text="  hello world  ")
        assert post.preprocess_text() == "hello world"

    def test_empty_text_returns_empty(self) -> None:
        post = make_post(post_text="")
        assert post.preprocess_text() == ""
