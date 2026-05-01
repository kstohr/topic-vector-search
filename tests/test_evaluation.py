from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    ComputePrecisionResp,
    ComputeRecallResp,
    SearchResultRow,
    SearchResultRowsArgs,
    TopicEvalArgs,
    TopicSearchMetricArgs,
    build_search_result_rows,
    compute_precision_at_k,
    compute_random_baseline,
    compute_recall_at_k,
    compute_topic_search_eval_metrics,
    evaluate_topics,
)


def test_compute_random_baseline_returns_topic_fraction() -> None:
    assert compute_random_baseline(topic_size=25, dataset_size=100) == 0.25


def test_compute_random_baseline_returns_zero_when_dataset_empty() -> None:
    assert compute_random_baseline(topic_size=5, dataset_size=0) == 0.0


def test_compute_precision_at_k_counts_hits_in_top_k() -> None:
    result = compute_precision_at_k(
        retrieved_ids=["p1", "p2", "p3", "p4"],
        topic_post_ids={"p2", "p4", "p9"},
        k=3,
    )

    assert result.hits == 1
    assert result.precision == pytest.approx(1 / 3)


def test_compute_precision_at_k_uses_all_retrieved_if_less_than_k() -> None:
    result = compute_precision_at_k(
        retrieved_ids=["p1", "p2"],
        topic_post_ids={"p1", "p3"},
        k=10,
    )

    assert result.hits == 1
    assert result.precision == pytest.approx(1 / 2)


def test_compute_precision_at_k_returns_zero_for_empty_retrieval() -> None:
    result = compute_precision_at_k(
        retrieved_ids=[],
        topic_post_ids={"p1", "p2"},
        k=5,
    )

    assert result == ComputePrecisionResp(precision=0.0, hits=0)


def test_compute_recall_at_k_counts_hits_against_all_relevant_posts() -> None:
    result = compute_recall_at_k(
        retrieved_ids=["a", "b", "c", "d"],
        topic_post_ids={"b", "d", "z", "y"},
        k=3,
    )

    assert result.hits == 1
    assert result.recall == pytest.approx(1 / 4)


def test_compute_recall_at_k_returns_zero_for_empty_topic_posts() -> None:
    result = compute_recall_at_k(
        retrieved_ids=["a", "b", "c"],
        topic_post_ids=set(),
        k=3,
    )

    assert result == ComputeRecallResp(recall=0.0, hits=0)


def test_compute_recall_at_k_uses_top_k_only() -> None:
    result = compute_recall_at_k(
        retrieved_ids=["hit1", "miss", "hit2", "hit3"],
        topic_post_ids={"hit1", "hit2", "hit3"},
        k=2,
    )

    assert result.hits == 1
    assert result.recall == pytest.approx(1 / 3)


def test_compute_topic_search_eval_metrics_counts_precision_recall_and_baseline() -> None:
    args = TopicSearchMetricArgs(
        dataset_size=10,
        retrieved_ids=["a", "b", "x", "c", "d"],
        topic_post_ids={"a", "b", "c", "z"},
        precision_k=3,
        recall_k=5,
    )

    result = compute_topic_search_eval_metrics(args)

    assert result.precision_hits == 2
    assert result.precision_at_k == pytest.approx(2 / 3)
    assert result.recall_hits == 3
    assert result.recall_at_k == pytest.approx(3 / 4)
    assert result.baseline == pytest.approx(4 / 10)
    assert result.num_posts_assigned_to_topic == 4
    assert result.num_retrieved_by_search == 5


def test_evaluate_topics_returns_one_row_per_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    search_results_by_embedding = {
        (1.0, 0.0): [
            {"post_id": "p1", "score": 0.9},
            {"post_id": "p2", "score": 0.8},
            {"post_id": "p999", "score": 0.1},
        ],
        (0.0, 1.0): [
            {"post_id": "p3", "score": 0.7},
            {"post_id": "p1", "score": 0.2},
        ],
    }

    def fake_run_search_by_topic(args):
        assert hasattr(args.searcher, "embedding_model")
        assert args.top_k == 5
        key = tuple(float(x) for x in args.embedding.tolist())
        return search_results_by_embedding[key]

    monkeypatch.setattr(
        "src.evaluation.run_search_by_topic",
        fake_run_search_by_topic,
    )

    # callable searcher with required attribute
    def fake_searcher(query_embedding, top_k):
        return []

    fake_searcher.embedding_model = "fake-model"

    args = TopicEvalArgs(
        corpus_size=4,
        labels={
            1: {"label": "Alpha"},
            2: {"label": "Beta"},
        },
        assignments=pd.DataFrame(
            {
                "topic_id": [1, 1, 2, 2],
                "post_id": ["p1", "p2", "p3", "p4"],
            }
        ),
        topic_embeddings={1: [1.0, 0.0], 2: [0.0, 1.0]},
        searcher=fake_searcher,
        keywords={1: ["one", "two"], 2: ["three"]},
        eval_k=5,
    )

    df = evaluate_topics(args)

    assert list(df["topic"]) == ["Alpha", "Beta"]
    assert list(df["keywords"]) == ["one, two", "three"]
    assert list(df["assigned_posts"]) == [2, 2]
    assert df.loc[0, "avg_search_score"] == pytest.approx(np.mean([0.9, 0.8, 0.1]))
    assert df.loc[1, "avg_search_score"] == pytest.approx(np.mean([0.7, 0.2]))
    assert df.loc[0, "recall_hits"] == 2
    assert df.loc[1, "recall_hits"] == 1


def test_build_search_result_rows_prefers_result_text_and_joins_topic() -> None:
    rows = build_search_result_rows(
        SearchResultRowsArgs(
            results=[{"post_id": "p1", "post_text": " Search result text ", "score": 0.87654}],
            posts_by_id={
                "p1": {"image_caption": "caption", "image_url": "https://example.com/img.jpg"}
            },
            assignments=pd.DataFrame({"post_id": ["p1"], "topic_id": [7]}),
            labels={7: {"label": "Topic Seven"}},
        )
    )

    assert rows == [
        SearchResultRow(
            score=0.877,
            post="Search result text",
            topic="Topic Seven",
            image_url="https://example.com/img.jpg",
            post_id="p1",
        )
    ]


def test_build_search_result_rows_uses_caption_when_text_is_missing() -> None:
    rows = build_search_result_rows(
        SearchResultRowsArgs(
            results=[{"post_id": "p2", "post_text": "", "score": 0.5}],
            posts_by_id={"p2": {"image_caption": "a useful caption", "image_url": ""}},
            assignments=pd.DataFrame({"post_id": ["p2"], "topic_id": [99]}),
            labels={},
        )
    )

    assert rows[0].post == "[image] a useful caption"
    assert rows[0].topic == "Topic 99"


def test_build_search_result_rows_uses_placeholder_for_image_without_caption() -> None:
    rows = build_search_result_rows(
        SearchResultRowsArgs(
            results=[{"post_id": "missing", "score": 0.1}],
            posts_by_id={},
            assignments=pd.DataFrame({"post_id": [], "topic_id": []}),
            labels={},
        )
    )

    assert rows[0].post == "[image — no caption yet]"
    assert rows[0].topic == ""
    assert rows[0].image_url == ""
