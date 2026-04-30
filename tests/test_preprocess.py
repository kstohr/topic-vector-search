from __future__ import annotations

from src.preprocess import embedding_text

BASE_POST = {
    "post_id": "x",
    "post_author": "u",
    "created_at": "2024-01-01T00:00:00",
    "modified_at": "2024-01-01T00:00:00",
    "post_text": "hello world",
    "likes": 0,
}


def make(post_text: str = "hello world", image_caption: str | None = None) -> dict:
    return {**BASE_POST, "post_text": post_text, "image_caption": image_caption}


def test_text_only_returns_preprocessed_text() -> None:
    result = embedding_text(make("Hello World!"))
    assert result == "hello world"


def test_caption_only_returns_caption() -> None:
    result = embedding_text(make(post_text="", image_caption="a cute cat"))
    assert result == "a cute cat"


def test_both_text_and_caption_combined() -> None:
    result = embedding_text(make("hello world", image_caption="a cute cat"))
    assert result == "hello world a cute cat"


def test_no_text_no_caption_returns_empty() -> None:
    result = embedding_text(make(post_text="", image_caption=None))
    assert result == ""


def test_whitespace_only_text_treated_as_empty() -> None:
    result = embedding_text(make(post_text="   ", image_caption="caption"))
    assert result == "caption"
