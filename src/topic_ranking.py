from datetime import datetime, timezone

import pandas as pd


def rank_topics(
    assignments: pd.DataFrame,
    posts_by_id: dict[str, dict],
    labels: dict[int, dict],
    keywords: dict[int, list[str]],
    top_n: int = 3,
) -> list[dict]:
    """Score topics by engagement and recency, return the top N trending."""

    def parse_ts(raw: str) -> float:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()

    most_recent_ts = max(parse_ts(p["created_at"]) for p in posts_by_id.values())

    rows = []
    for topic_id in sorted(labels.keys()):
        post_ids = assignments.loc[assignments["topic_id"] == topic_id, "post_id"].tolist()
        if not post_ids:
            continue
        total_likes = sum(posts_by_id.get(pid, {}).get("likes", 0) for pid in post_ids)
        ts_list = [
            parse_ts(posts_by_id[pid]["created_at"])
            for pid in post_ids if pid in posts_by_id
        ]
        avg_days_ago = (most_recent_ts - sum(ts_list) / len(ts_list)) / 86400 if ts_list else 999
        recency = max(0.0, 100.0 - (avg_days_ago / 365) * 100)
        trending_score = total_likes * 0.65 + recency * 0.35

        info = labels[topic_id]
        rows.append({
            "topic_id": topic_id,
            "label": info["label"],
            "emoji": info["emoji"],
            "total_likes": total_likes,
            "post_count": len(post_ids),
            "trending_score": trending_score,
            "keywords": keywords.get(topic_id, [])[:5],
        })

    rows.sort(key=lambda r: r["trending_score"], reverse=True)
    return rows[:top_n]
