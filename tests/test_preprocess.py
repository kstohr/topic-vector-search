from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data_models import PostDocument
from src.preprocess import PreprocessingPipeline, extract_embedding_text

BASE_POST = {
    "post_id": "x",
    "post_author": "u",
    "created_at": "2024-01-01T00:00:00",
    "modified_at": "2024-01-01T00:00:00",
    "post_text": "hello world",
    "likes": 0,
}


def make(post_text: str = "hello world", image_caption: str | None = None) -> PostDocument:
    return PostDocument(**{**BASE_POST, "post_text": post_text, "image_caption": image_caption})


def make_with_image(image_url: str | None) -> PostDocument:
    return PostDocument(
        **{
            **BASE_POST,
            "post_text": "",
            "image_url": image_url,
            "image_caption": None,
        }
    )


class _FakeVisionInputs(dict):
    def to(self, _device: str):
        return self


class _FakeVisionProcessor:
    def __init__(self) -> None:
        self.called = False

    def __call__(self, *, images, return_tensors: str):
        self.called = True
        assert images is not None
        assert return_tensors == "pt"
        return _FakeVisionInputs({"pixel_values": "fake-tensor"})

    def decode(self, _output_ids, skip_special_tokens: bool):
        assert skip_special_tokens is True
        return "a caption from fake model"


class _FakeVisionModel:
    def __init__(self) -> None:
        self.called = False

    def generate(self, **kwargs):
        self.called = True
        assert "pixel_values" in kwargs
        return [[101, 102, 103]]


class _FakeEmbeddingModel:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors
        self.called = False
        self.last_texts: list[str] | None = None

    def encode(self, texts, **_kwargs):
        self.called = True
        self.last_texts = list(texts)
        return np.array(self._vectors, dtype=np.float32)


def _make_pipeline_with_fakes() -> tuple[
    PreprocessingPipeline, _FakeVisionProcessor, _FakeVisionModel
]:
    # Avoid heavy model downloads in tests by creating an uninitialized instance.
    pipeline = PreprocessingPipeline.__new__(PreprocessingPipeline)
    processor = _FakeVisionProcessor()
    model = _FakeVisionModel()
    pipeline._vision_processor = processor
    pipeline._vision_model = model
    return pipeline, processor, model


def _make_embedding_pipeline(
    vectors: list[list[float]],
) -> tuple[PreprocessingPipeline, _FakeEmbeddingModel]:
    pipeline = PreprocessingPipeline.__new__(PreprocessingPipeline)
    fake_embedding_model = _FakeEmbeddingModel(vectors=vectors)
    pipeline.embedding_model = fake_embedding_model
    return pipeline, fake_embedding_model


def test_text_only_returns_preprocessed_text() -> None:
    result = extract_embedding_text(make("Hello World!"))
    assert result == "hello world"


def test_caption_only_returns_caption() -> None:
    result = extract_embedding_text(make(post_text="", image_caption="a cute cat"))
    assert result == "a cute cat"


def test_both_text_and_caption_combined() -> None:
    result = extract_embedding_text(make("hello world", image_caption="a cute cat"))
    assert result == "hello world a cute cat"


def test_no_text_no_caption_returns_empty() -> None:
    result = extract_embedding_text(make(post_text="", image_caption=None))
    assert result == ""


def test_whitespace_only_text_treated_as_empty() -> None:
    result = extract_embedding_text(make(post_text="   ", image_caption="caption"))
    assert result == "caption"


def test_caption_single_post_returns_early_when_no_image_url() -> None:
    pipeline, processor, model = _make_pipeline_with_fakes()
    postdoc = make_with_image(image_url=None)

    pipeline._caption_single_post(postdoc)

    assert postdoc.image_caption is None
    assert processor.called is False
    assert model.called is False


def test_caption_single_post_returns_when_image_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pipeline, processor, model = _make_pipeline_with_fakes()
    postdoc = make_with_image(image_url="missing.jpg")
    monkeypatch.setattr("src.preprocess.REPO", tmp_path)

    pipeline._caption_single_post(postdoc)

    assert postdoc.image_caption is None
    assert processor.called is False
    assert model.called is False


@pytest.mark.exercise
def test_caption_single_post_sets_caption_from_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pipeline, processor, model = _make_pipeline_with_fakes()
    image_name = "cat.jpg"
    image_path = tmp_path / image_name
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
    postdoc = make_with_image(image_url=image_name)
    monkeypatch.setattr("src.preprocess.REPO", tmp_path)

    pipeline._caption_single_post(postdoc)

    assert processor.called is True
    assert model.called is True
    assert postdoc.image_caption == "a caption from fake model"


@pytest.mark.exercise
def test_generate_embeddings_stores_embedding_vectors_on_posts() -> None:
    pipeline, fake_embedding_model = _make_embedding_pipeline(vectors=[[0.1, 0.2], [0.3, 0.4]])
    postdocs = [
        make(post_text="Hello world", image_caption="A cat"),
        make(post_text="", image_caption="only caption"),
    ]

    result = pipeline.generate_embeddings(postdocs)

    assert result is postdocs
    assert fake_embedding_model.called is True
    assert fake_embedding_model.last_texts == ["hello world A cat", "only caption"]
    assert np.allclose(postdocs[0].doc_embedding, [0.1, 0.2])
    assert np.allclose(postdocs[1].doc_embedding, [0.3, 0.4])


@pytest.mark.exercise
def test_generate_embeddings_handles_empty_posts_list() -> None:
    pipeline, fake_embedding_model = _make_embedding_pipeline(vectors=[])
    postdocs: list[PostDocument] = []

    result = pipeline.generate_embeddings(postdocs)

    assert result == []
    assert fake_embedding_model.called is True
    assert fake_embedding_model.last_texts == []
