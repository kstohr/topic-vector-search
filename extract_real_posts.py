"""
Extract real posts from the production post_index.json and save them in
sample_posts format as sample_posts_expanded.json.

Embeddings are already computed with all-MiniLM-L6-v2 (same model as the
workshop), so no re-embedding step is needed — run src.preprocess to index
them into Elasticsearch, then src.topic_model to train.

Usage:
    uv run python extract_real_posts.py
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

POST_INDEX = Path(
    "/Users/kas/dev/unf/ufd_notebooks/ufd_notebooks/projects/topic_modeling"
    "/output/bertopic_base/v0.1/2024-08-06"
    "/topic_model/bertopic_base/v0.1/2024-08-06"
    "/data/post_index.json"
)
OUT = Path("sample_posts_2024.json")

MIN_TEXT_LEN = 20


def _normalise_ts(raw: str) -> str:
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.astimezone(UTC).isoformat()
    except Exception:
        return datetime.now(UTC).isoformat()


def _is_usable(text: str) -> bool:
    if not text or len(text.strip()) < MIN_TEXT_LEN:
        return False
    # Skip posts that are only a URL
    if re.fullmatch(r"https?://\S+", text.strip()):
        return False
    return True


def main() -> None:
    with open(POST_INDEX) as f:
        raw = json.load(f)

    posts = []
    skipped = 0
    for pid, doc in raw.items():
        body = doc.get("body") or {}
        if isinstance(body, str):
            try:
                import ast
                body = ast.literal_eval(body)
            except Exception:
                body = {}

        text = (body.get("post_text") or "").strip()
        if not _is_usable(text):
            skipped += 1
            continue

        posts.append({
            "post_id":        doc["post_id"],
            "post_author":    str(doc.get("post_author", "unknown")),
            "created_at":     _normalise_ts(str(doc.get("created_at", ""))),
            "modified_at":    _normalise_ts(str(doc.get("modified_at", ""))),
            "post_text":      text,
            "likes":          0,
            "generated_topic": None,
            "txt_embedding":  [],
            "doc_embedding":  doc.get("doc_embedding") or [],
        })

    OUT.write_text(json.dumps(posts, indent=2, ensure_ascii=False))
    print(f"Extracted {len(posts)} posts ({skipped} skipped) → {OUT}")
    print(f"Posts with embeddings: {sum(1 for p in posts if p['doc_embedding'])}")


if __name__ == "__main__":
    main()
