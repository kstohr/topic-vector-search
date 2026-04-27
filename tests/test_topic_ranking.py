from __future__ import annotations

import pandas as pd
import pytest

from src.topic_ranking import rank_topics

LABELS = {
    1: {"label": "Cats", "emoji": "🐱"},
    2: {"label": "Rails", "emoji": "🚄"},
    3: {"label": "Space", "emoji": "🚀"},
}

KEYWORDS = {1: ["cat", "meow"], 2: ["rail", "train"], 3: ["rocket", "nasa"]}

ASSIGNMENTS = pd.DataFrame({
    "topic_id": [1, 1, 2, 3],
    "post_id":  ["p1", "p2", "p3", "p4"],
})

POSTS = {
    "p1": {"likes": 100, "created_at": "2024-06-01T00:00:00"},
    "p2": {"likes": 50,  "created_at": "2024-06-02T00:00:00"},
    "p3": {"likes": 10,  "created_at": "2024-06-03T00:00:00"},
    "p4": {"likes": 5,   "created_at": "2024-06-04T00:00:00"},
}


def test_returns_at_most_top_n() -> None:
    result = rank_topics(ASSIGNMENTS, POSTS, LABELS, KEYWORDS, top_n=2)
    assert len(result) == 2


def test_returns_all_when_fewer_than_top_n() -> None:
    result = rank_topics(ASSIGNMENTS, POSTS, LABELS, KEYWORDS, top_n=10)
    assert len(result) == 3


def test_result_contains_expected_keys() -> None:
    result = rank_topics(ASSIGNMENTS, POSTS, LABELS, KEYWORDS, top_n=1)
    row = result[0]
    for key in ("topic_id", "label", "emoji", "total_likes", "post_count", "trending_score", "keywords"):
        assert key in row


def test_total_likes_sum_correctly() -> None:
    result = rank_topics(ASSIGNMENTS, POSTS, LABELS, KEYWORDS, top_n=3)
    by_id = {r["topic_id"]: r for r in result}
    assert by_id[1]["total_likes"] == 150
    assert by_id[2]["total_likes"] == 10
    assert by_id[3]["total_likes"] == 5


def test_post_count_matches_assignments() -> None:
    result = rank_topics(ASSIGNMENTS, POSTS, LABELS, KEYWORDS, top_n=3)
    by_id = {r["topic_id"]: r for r in result}
    assert by_id[1]["post_count"] == 2
    assert by_id[2]["post_count"] == 1


def test_keywords_capped_at_five() -> None:
    long_keywords = {1: ["a", "b", "c", "d", "e", "f", "g"]}
    result = rank_topics(ASSIGNMENTS, POSTS, {1: LABELS[1]}, long_keywords, top_n=1)
    assert len(result[0]["keywords"]) == 5


def test_topic_with_no_posts_excluded() -> None:
    labels_with_empty = {**LABELS, 99: {"label": "Empty", "emoji": "❓"}}
    result = rank_topics(ASSIGNMENTS, POSTS, labels_with_empty, KEYWORDS, top_n=10)
    topic_ids = [r["topic_id"] for r in result]
    assert 99 not in topic_ids


def test_highest_engagement_topic_ranks_first() -> None:
    result = rank_topics(ASSIGNMENTS, POSTS, LABELS, KEYWORDS, top_n=3)
    assert result[0]["topic_id"] == 1
