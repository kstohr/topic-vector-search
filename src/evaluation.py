import numpy as np
import pandas as pd


def build_result_rows(
    results: list[dict],
    posts_by_id: dict[str, dict],
    assignments: pd.DataFrame,
    labels: dict[int, dict],
) -> list[dict]:
    """
    Join search results with topic assignments and post metadata.
    Returns a list of flat dicts ready for display or ranking.
    """
    id_to_label = {tid: f"{v['emoji']} {v['label']}" for tid, v in labels.items()}

    rows = []
    for r in results:
        pid = r.get("post_id", "")
        post = posts_by_id.get(pid, {})
        text = r.get("post_text", "").strip()
        caption = (post.get("image_caption") or "").strip()

        if text:
            display_text = text
        elif caption:
            display_text = f"[image] {caption}"
        else:
            display_text = "[image — no caption yet]"

        topic_label = ""
        match = assignments.loc[assignments["post_id"] == pid, "topic_id"]
        if len(match):
            tid = int(match.values[0])
            topic_label = id_to_label.get(tid, f"Topic {tid}")

        rows.append({
            "score": round(r.get("score", 0), 3),
            "post": display_text,
            "topic": topic_label,
            "image_url": post.get("image_url", ""),
            "post_id": pid,
        })

    return rows


def evaluate_topics(
    labels: dict[int, dict],
    assignments: pd.DataFrame,
    topic_centroid_embeddings: dict[int, list[float]],
    searcher,
    keywords: dict[int, list[str]] | None = None,
    top_n: int = 8,
) -> pd.DataFrame:
    """
    For each topic, run a semantic search using its embedding and compute
    precision metrics against the ground-truth topic assignments.

    Search size matches the number of posts assigned to each topic (no fixed
    cap), so match_ratio is precision@assigned_count — the fraction of assigned
    posts that appear when the search returns exactly that many results.

    Columns returned:
      topic           — label with emoji
      keywords        — top keywords by c-TF-IDF score, computed during BERTopic training
      assigned_posts  — posts assigned to this topic by the model
      results_matched — how many results belong to this topic
      match_ratio     — results_matched / assigned_posts  (precision@assigned_count)
      match_ratio_8   — matched in first top_n results / top_n
      median_score    — median similarity score across all results
      median_score_8  — median similarity score for first top_n results
    """
    from src.search import run_semantic_search

    assigned_by_topic = {
        tid: set(assignments.loc[assignments["topic_id"] == tid, "post_id"])
        for tid in labels
    }

    rows = []
    for topic_id, info in sorted(labels.items()):
        emb = np.array(topic_centroid_embeddings[topic_id], dtype=np.float32)
        assigned = assigned_by_topic.get(topic_id, set())

        top_k = len(assigned)
        results = run_semantic_search(emb, searcher, top_k=top_k)

        scores = [r["score"] for r in results]
        matched = [r for r in results if r.get("post_id") in assigned]
        top_n_results = results[:top_n]
        matched_top_n = [r for r in top_n_results if r.get("post_id") in assigned]

        kws = keywords.get(topic_id, [])[:10] if keywords else []

        rows.append({
            "topic": f"{info['emoji']} {info['label']}",
            "keywords": ", ".join(kws),
            "assigned_posts": len(assigned),
            "results_matched": len(matched),
            "match_ratio": round(len(matched) / len(assigned), 3) if assigned else 0.0,
            "match_ratio_8": round(len(matched_top_n) / top_n, 3) if top_n_results else 0.0,
            "median_score": round(float(np.median(scores)), 3) if scores else 0.0,
            "median_score_8": round(float(np.median([r["score"] for r in top_n_results])), 3) if top_n_results else 0.0,
        })

    return pd.DataFrame(rows)
