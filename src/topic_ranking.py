"""
====================
TOPIC RANKING
====================
Score and rank topics by engagement (likes) and recency.
"""

from datetime import UTC, datetime

import pandas as pd
from pydantic import BaseModel, ConfigDict


class TrendingTopic(BaseModel):
    """A ranked topic with engagement and recency scores."""

    topic_id: int
    label: str
    total_likes: int
    post_count: int
    trending_score: float
    keywords: list[str]


class TopicRankingArgs(BaseModel):
    """Input arguments for rank_topics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    assignments: pd.DataFrame
    posts_by_id: dict[str, dict]
    labels: dict[int, dict]
    keywords: dict[int, list[str]]
    top_n: int = 3


def _parse_ts(raw: str) -> float:
    """Parse an ISO datetime string into a UTC Unix timestamp."""
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:  # noqa: SIM108
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def rank_topics(args: TopicRankingArgs) -> list[TrendingTopic]:
    """Score topics by engagement and recency, return the top N trending."""
    most_recent_ts = max(_parse_ts(p["created_at"]) for p in args.posts_by_id.values())

    rows = []
    for topic_id in sorted(args.labels.keys()):
        # get posts assigned to this topic
        post_ids = args.assignments.loc[
            args.assignments["topic_id"] == topic_id, "post_id"
        ].tolist()
        if not post_ids:
            continue

        # Sum likes to measure engagement
        total_likes = sum(args.posts_by_id.get(post_id, {}).get("likes", 0) for post_id in post_ids)
        timestamps = [
            _parse_ts(args.posts_by_id[post_id]["created_at"])
            for post_id in post_ids
            if post_id in args.posts_by_id
        ]

        # Calculate recency
        avg_days_ago = (
            (most_recent_ts - sum(timestamps) / len(timestamps)) / 86400 if timestamps else 999
        )
        recency = max(0.0, 100.0 - (avg_days_ago / 365) * 100)

        # Calculate a simple trending score with 65% weight on likes and 35% on recency
        trending_score = total_likes * 0.65 + recency * 0.35

        info = args.labels[topic_id]
        rows.append(
            TrendingTopic(
                topic_id=topic_id,
                label=info["label"],
                total_likes=total_likes,
                post_count=len(post_ids),
                trending_score=trending_score,
                keywords=args.keywords.get(topic_id, [])[:5],
            )
        )

    rows.sort(key=lambda topic: topic.trending_score, reverse=True)
    return rows[: args.top_n]
