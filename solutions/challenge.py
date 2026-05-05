"""Challenge helper: find the topic with the largest centroid-vs-keyword distance."""

import json

import numpy as np
import pandas as pd

from src.config import OUTPUT


def topic_embedding_distance_table(
    topic_embeddings: dict[str, list[float]],
    keyword_embeddings: dict[str, list[float]],
    labels: dict[str, dict],
) -> pd.DataFrame:
    """Return a DataFrame of centroid-vs-keyword cosine distance for each shared topic."""
    rows = []
    shared_topic_ids = sorted(
        set(topic_embeddings.keys()) & set(keyword_embeddings.keys()),
        key=int,
    )

    for topic_id in shared_topic_ids:
        topic_vec = np.array(topic_embeddings[topic_id], dtype=np.float32)
        keyword_vec = np.array(keyword_embeddings[topic_id], dtype=np.float32)

        denom = float(np.linalg.norm(topic_vec) * np.linalg.norm(keyword_vec))
        if denom == 0.0:
            continue

        cosine_similarity = float(np.dot(topic_vec, keyword_vec) / denom)
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
        distance = 1.0 - cosine_similarity

        rows.append(
            {
                "topic_id": int(topic_id),
                "label": labels.get(topic_id, {}).get("label", "<missing label>"),
                "cosine_similarity": cosine_similarity,
                "distance": distance,
            }
        )

    if not rows:
        raise ValueError("No comparable topics found between topic and keyword embeddings")

    return pd.DataFrame(rows).sort_values("distance", ascending=False).reset_index(drop=True)


def load_artifacts(
    output_dir=OUTPUT,
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, dict]]:
    """Load topic embeddings, keyword embeddings, and labels from the output directory."""
    topic_embeddings = json.loads((output_dir / "topic_embeddings.json").read_text())
    keyword_embeddings = json.loads((output_dir / "topic_keyword_embeddings.json").read_text())
    labels = json.loads((output_dir / "topic_labels.json").read_text())
    return topic_embeddings, keyword_embeddings, labels


def run_challenge() -> pd.DataFrame:
    """Compute and return ranked topic distances for the challenge prompt."""
    topic_embeddings, keyword_embeddings, labels = load_artifacts()
    return topic_embedding_distance_table(topic_embeddings, keyword_embeddings, labels)


if __name__ == "__main__":
    df = run_challenge()
    print("Top topic by centroid-vs-keyword distance:")
    print(df.head(1).to_string(index=False))

    print("\nAll topics by distance:")
    print(df.to_string(index=False))
