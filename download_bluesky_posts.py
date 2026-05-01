"""
Download a random sample of Bluesky posts and save them in sample_posts format.

Usage:
    uv run python download_bluesky_posts.py [--n 1000] [--out sample_posts_expanded.json]

Streams alpindale/two-million-bluesky-posts from Hugging Face without
downloading the full 828 MB dataset. Filters for English-like posts with
meaningful text, then randomly samples n posts.

Output format matches sample_posts.json — ready to be merged with or used
as a replacement for the synthetic dataset.
"""

import argparse
import json
import random
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

MIN_TEXT_LEN = 30
MAX_TEXT_LEN = 500
STREAM_LIMIT = 50_000  # posts to scan before sampling
RANDOM_SEED = 42


def _looks_english(text: str) -> bool:
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / max(len(text), 1) > 0.7


def _is_usable(post: dict) -> bool:
    text = (post.get("text") or "").strip()
    if not text:
        return False
    if len(text) < MIN_TEXT_LEN or len(text) > MAX_TEXT_LEN:
        return False
    # Skip posts that are pure URLs
    if re.fullmatch(r"https?://\S+", text):
        return False
    return _looks_english(text)


def _to_post_doc(post: dict) -> dict:
    uri = post.get("uri", "")
    created_raw = post.get("created_at") or datetime.now(UTC).isoformat()

    # Normalise timestamp to UTC ISO string
    try:
        dt = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        created_iso = dt.astimezone(UTC).isoformat()
    except Exception:
        created_iso = datetime.now(UTC).isoformat()

    return {
        "post_id": str(uuid.uuid5(uuid.NAMESPACE_URL, uri)) if uri else str(uuid.uuid4()),
        "post_author": post.get("author") or "unknown",
        "created_at": created_iso,
        "modified_at": created_iso,
        "post_text": post["text"].strip(),
        "likes": 0,
        "generated_topic": None,
        "txt_embedding": [],
        "doc_embedding": [],
    }


def download(n: int, out_path: Path) -> None:
    from datasets import load_dataset

    print(f"Streaming alpindale/two-million-bluesky-posts (scanning up to {STREAM_LIMIT:,} posts)…")
    ds = load_dataset(
        "alpindale/two-million-bluesky-posts",
        split="train",
        streaming=True,
    )

    candidates: list[dict] = []
    scanned = 0

    for raw in ds:
        scanned += 1
        if scanned > STREAM_LIMIT:
            break
        if _is_usable(raw):
            candidates.append(raw)
        if scanned % 5_000 == 0:
            print(f"  scanned {scanned:,} posts, {len(candidates):,} usable so far…")

    print(f"Scanned {scanned:,} posts total, {len(candidates):,} passed filters.")

    random.seed(RANDOM_SEED)
    sampled = random.sample(candidates, min(n, len(candidates)))
    posts = [_to_post_doc(p) for p in sampled]

    out_path.write_text(json.dumps(posts, indent=2, ensure_ascii=False))
    print(f"Saved {len(posts)} posts → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="Number of posts to sample")
    parser.add_argument("--out", type=Path, default=Path("sample_posts_expanded.json"))
    args = parser.parse_args()
    download(args.n, args.out)
