"""
Preprocessing pipeline: caption images, generate embeddings, store posts.

Run:
    uv run python -m src.preprocess

Steps:
  1. Load posts from sample_posts.json
  2. Caption any image-only posts that lack a caption (BLIP vision model)
  3. Batch-embed all posts (combines post_text + image_caption when both exist)
  4. Save to Elasticsearch (if running) and to output/processed_posts.json
  5. Write updated embeddings/captions back to sample_posts.json
"""

import json
import logging
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from src.models import PostDocument
from src.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME, OUTPUT, REPO, VISION_MODEL_NAME

logger = logging.getLogger(__name__)


# ── Text helper ────────────────────────────────────────────────────────────

def embedding_text(post: dict) -> str:
    """
    Build the text string passed to the embedding model.
    Combines post_text (preprocessed) with image_caption so that image-only
    posts are searchable via their visual content after LO5 captioning.
    """
    doc = PostDocument(**post)
    text = doc.preprocess_text().strip()
    caption = (post.get("image_caption") or "").strip()
    if text and caption:
        return f"{text} {caption}"
    return text or caption


# ── Image captioning (LO5) ─────────────────────────────────────────────────

def _load_blip() -> tuple[Any, Any, str]:
    """Load BLIP processor and model. Returns (processor, model, device)."""
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    model_name = VISION_MODEL_NAME
    logger.info(f"Loading vision model {model_name}…")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = model.to(device)
    logger.info(f"Vision model loaded on {device}.")
    return processor, model, device


def caption_images(posts: list[dict]) -> list[dict]:
    """
    Caption image-only posts that have no caption yet.
    Silently skips if transformers / PIL are not installed.
    """
    needs_caption = [
        p for p in posts if p.get("image_url") and not p.get("image_caption")
    ]
    if not needs_caption:
        logger.info("No image posts need captioning.")
        return posts

    try:
        from PIL import Image
        processor, model, device = _load_blip()
    except Exception as e:
        logger.warning(f"Vision model unavailable ({e}) — skipping image captioning.")
        return posts

    for p in needs_caption:
        img_path = REPO / p["image_url"]
        if not img_path.exists():
            logger.warning(f"Image file not found: {img_path}")
            continue
        logger.info(f"Captioning {img_path.name}…")
        image = Image.open(img_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        p["image_caption"] = caption
        logger.info(f"  → {caption}")

    return posts


# ── Embedding ──────────────────────────────────────────────────────────────

def generate_embeddings(posts: list[dict], model: SentenceTransformer) -> list[dict]:
    """
    Embed all posts that need it.
    Re-embeds image posts that now have a caption but were previously embedded
    without one (their old embedding was based on empty text).
    """
    for p in posts:
        if p.get("image_url") and p.get("image_caption") and p.get("doc_embedding"):
            if not p.get("post_text", "").strip():
                p["doc_embedding"] = []  # force re-embed with caption text

    needs = [p for p in posts if not p.get("doc_embedding")]
    logger.info(f"Embedding {len(needs)} posts ({len(posts) - len(needs)} already done)…")

    if not needs:
        return posts

    texts = [embedding_text(p) for p in needs]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True,
                              convert_to_numpy=True)
    for p, emb in zip(needs, embeddings):
        p["doc_embedding"] = emb.tolist()

    return posts


# ── Storage ────────────────────────────────────────────────────────────────

def _try_elasticsearch_client() -> Elasticsearch | None:
    try:
        client = Elasticsearch("http://localhost:9200")
        client.info()
        return client
    except Exception:
        return None


def save_to_elasticsearch(posts: list[dict], client: Elasticsearch, index_name: str = "post_docs") -> None:
    from src.index import INDEX_NAME, create_index

    create_index(client)
    for p in posts:
        doc = PostDocument(**p)
        client.index(index=INDEX_NAME, id=doc.post_id, body=doc.model_dump(mode="json"))
    logger.info(f"Stored {len(posts)} posts in Elasticsearch index '{index_name}'.")


def save_to_disk(posts: list[dict]) -> None:
    OUTPUT.mkdir(exist_ok=True)

    doc_index = {p["post_id"]: p for p in posts}
    with open(OUTPUT / "processed_posts.json", "w") as f:
        json.dump(doc_index, f, default=str)
    logger.info(f"Saved processed_posts.json ({len(doc_index)} posts).")

    with open(REPO / "sample_posts.json", "w") as f:
        json.dump(posts, f, indent=2, default=str)
    logger.info("Updated sample_posts.json.")


# ── Entry point ────────────────────────────────────────────────────────────

def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    logger.info("Loading sample_posts.json…")
    with open(REPO / "sample_posts.json") as f:
        posts = json.load(f)

    posts = caption_images(posts)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    posts = generate_embeddings(posts, model)

    client = _try_elasticsearch_client()
    if client:
        logger.info("Elasticsearch available — saving posts.")
        save_to_elasticsearch(posts, client)
    else:
        logger.info("Elasticsearch not available — using disk only.")

    save_to_disk(posts)
    logger.info("Preprocessing complete. Run: uv run python -m src.topic_model")


if __name__ == "__main__":
    run()
