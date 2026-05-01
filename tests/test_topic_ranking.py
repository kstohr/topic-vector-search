from __future__ import annotations

import pandas as pd

from src.topic_ranking import TopicRankingArgs, TrendingTopic, rank_topics

LABELS = {
    1: {"label": "Cats"},
    2: {"label": "Rails"},
    3: {"label": "Space"},
}

KEYWORDS = {1: ["cat", "meow"], 2: ["rail", "train"], 3: ["rocket", "nasa"]}

ASSIGNMENTS = pd.DataFrame(
    {
        "topic_id": [1, 1, 2, 3],
        "post_id": ["p1", "p2", "p3", "p4"],
    }
)

POSTS = {
    "p1": {"likes": 100, "created_at": "2024-06-01T00:00:00"},
    "p2": {"likes": 50, "created_at": "2024-06-02T00:00:00"},
    "p3": {"likes": 10, "created_at": "2024-06-03T00:00:00"},
    "p4": {"likes": 5, "created_at": "2024-06-04T00:00:00"},
}


def test_returns_at_most_top_n() -> None:
    result = rank_topics(
        TopicRankingArgs(
            assignments=ASSIGNMENTS, posts_by_id=POSTS, labels=LABELS, keywords=KEYWORDS, top_n=2
        )
    )
    assert len(result) == 2


def test_returns_all_when_fewer_than_top_n() -> None:
    result = rank_topics(
        TopicRankingArgs(
            assignments=ASSIGNMENTS, posts_by_id=POSTS, labels=LABELS, keywords=KEYWORDS, top_n=10
        )
    )
    assert len(result) == 3


def test_result_contains_expected_fields() -> None:
    result = rank_topics(
        TopicRankingArgs(
            assignments=ASSIGNMENTS, posts_by_id=POSTS, labels=LABELS, keywords=KEYWORDS, top_n=1
        )
    )
    assert isinstance(result[0], TrendingTopic)


def test_total_likes_sum_correctly() -> None:
    result = rank_topics(
        TopicRankingArgs(
            assignments=ASSIGNMENTS, posts_by_id=POSTS, labels=LABELS, keywords=KEYWORDS, top_n=3
        )
    )
    by_id = {r.topic_id: r for r in result}
    assert by_id[1].total_likes == 150
    assert by_id[2].total_likes == 10
    assert by_id[3].total_likes == 5


def test_post_count_matches_assignments() -> None:
    result = rank_topics(
        TopicRankingArgs(
            assignments=ASSIGNMENTS, posts_by_id=POSTS, labels=LABELS, keywords=KEYWORDS, top_n=3
        )
    )
    by_id = {r.topic_id: r for r in result}
    assert by_id[1].post_count == 2
    assert by_id[2].post_count == 1


def test_keywords_capped_at_five() -> None:
    long_keywords = {1: ["a", "b", "c", "d", "e", "f", "g"]}
    result = rank_topics(
        TopicRankingArgs(
            assignments=ASSIGNMENTS,
            posts_by_id=POSTS,
            labels={1: LABELS[1]},
            keywords=long_keywords,
            top_n=1,
        )
    )
    assert len(result[0].keywords) == 5


def test_topic_with_no_posts_excluded() -> None:
    labels_with_empty = {**LABELS, 99: {"label": "Empty"}}
    result = rank_topics(
        TopicRankingArgs(
            assignments=ASSIGNMENTS,
            posts_by_id=POSTS,
            labels=labels_with_empty,
            keywords=KEYWORDS,
            top_n=10,
        )
    )
    topic_ids = [r.topic_id for r in result]
    assert 99 not in topic_ids


def test_highest_engagement_topic_ranks_first() -> None:
    result = rank_topics(
        TopicRankingArgs(
            assignments=ASSIGNMENTS, posts_by_id=POSTS, labels=LABELS, keywords=KEYWORDS, top_n=3
        )
    )
    assert result[0].topic_id == 1
