"""
====================
GENERATE NOISE

Add posts that don't cluster well to any one topic and create "noise" in the
dataset. This will make the search retrieval task more realistic and challenging.
====================

Generates synthetic posts whose keywords are borrowed from *neighbor* topics
(near the target topic in embedding space) mixed with generic filler words.
The posts are plausible enough to be retrieved when searching by the target
topic embedding, but they were never assigned to that topic by BERTopic — so
they act as hard negatives when evaluating precision@K.

Run:
    uv run python -m src.generate_noise

Steps:
  1. Load the trained BERTopic model from output/bertopic_model/
  2. Load topic_embeddings.json; rank all topics by cosine similarity to the
     target topic; pick the top `num_neighbors` (always including the "cats"
     topic for workshop demo purposes)
  3. For each neighbor topic harvest its BERTopic keywords; generate
     `posts_per_neighbor` synthetic posts by mixing those keywords with random
     filler words
  4. Save the posts to output/noise_posts.json (same list format as
     sample_posts.json so PreprocessingPipeline can consume it)
  5. Run PreprocessingPipeline on the noise file to embed and index the posts
"""

import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from src.config import OUTPUT
from src.data_models import Post
from src.es_index import INDEX_NAME, get_es_client
from src.topic_model import TopicModeler

logger = logging.getLogger(__name__)

NOISE_OUTPUT_FILEPATH = OUTPUT / "noise_posts.json"

# Generic filler words used to pad synthetic post text so posts look natural
# but don't belong to any single topic.
FILLER_WORDS = [
    "amazing",
    "today",
    "love",
    "weekend",
    "post",
    "share",
    "everyone",
    "great",
    "just",
    "really",
    "so",
    "new",
    "got",
    "check",
    "out",
    "looking",
    "feel",
    "think",
    "like",
    "good",
]

# Topic 0 is "Cat Love and Bonds" — always include it for the workshop demo.
CAT_TOPIC_ID = 0


class NoiseGeneratorArgs(BaseModel):
    target_topic_id: int = 0
    num_neighbors: int = 5
    posts_per_neighbor: int = 25
    pipeline_output_path: Path = Path("output")


class NoiseGenerator:
    """Generate synthetic noise posts and index them into Elasticsearch."""

    def __init__(self, args: NoiseGeneratorArgs) -> None:
        self.args = args
        self.output_path = args.pipeline_output_path

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two vectors; returns 0.0 when either is zero-length."""
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        return float(np.dot(vec_a, vec_b) / denom) if denom > 0 else 0.0

    def _find_neighbor_topic_ids(
        self,
        topic_embeddings: dict[str, list[float]],
    ) -> list[int]:
        """Return the IDs of the `num_neighbors` topics closest to the target.

        The cats topic (CAT_TOPIC_ID) is always injected when the target is
        not already the cats topic, so the workshop demo always has it present.
        """
        target_id = self.args.target_topic_id
        if str(target_id) not in topic_embeddings:
            raise KeyError(
                f"Topic ID {target_id} not found in topic_embeddings.json. "
                "Run the full pipeline first: uv run python -m src.run_pipeline"
            )
        target_embedding = np.array(topic_embeddings[str(target_id)])

        scores: list[tuple[int, float]] = []
        for topic_id_str, topic_embedding in topic_embeddings.items():
            topic_id = int(topic_id_str)
            if topic_id == target_id:
                continue
            cosine_sim = self._cosine_similarity(target_embedding, np.array(topic_embedding))
            scores.append((topic_id, cosine_sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        neighbors = [topic_id for topic_id, _ in scores[: self.args.num_neighbors]]

        # Ensure cats topic is present when it isn't the target
        if target_id != CAT_TOPIC_ID and CAT_TOPIC_ID not in neighbors:
            neighbors[-1] = CAT_TOPIC_ID  # replace the weakest neighbor

        return neighbors

    @staticmethod
    def _generate_post_text(keywords: list[str]) -> str:
        """Mix 2–3 topic keywords with 4–6 generic filler words."""
        sampled_keywords = random.sample(keywords, min(random.randint(2, 3), len(keywords)))
        fillers = random.sample(FILLER_WORDS, random.randint(4, 6))
        parts = sampled_keywords + fillers
        random.shuffle(parts)
        return " ".join(parts)

    @staticmethod
    def _make_post(post_text: str, idx: int) -> Post:
        """Build a Post with a zero-padded noise-{idx:04d} ID and author 'noise'."""
        now = datetime.now(UTC).isoformat()
        return Post(
            post_id=f"noise-{idx:04d}",
            post_author="noise",
            created_at=now,
            modified_at=now,
            post_text=post_text,
            likes=0,
        )

    # ── Main pipeline ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Generate noise posts, write them to disk, and index them."""
        logger.info(
            f"Generating noise posts — target_topic_id={self.args.target_topic_id}, "
            f"num_neighbors={self.args.num_neighbors}, "
            f"posts_per_neighbor={self.args.posts_per_neighbor}"
        )

        # ── Fail fast: check required artifacts before loading anything expensive ──
        model_path = self.output_path / "bertopic_model"
        if not model_path.exists():
            raise FileNotFoundError(
                f"BERTopic model not found at {model_path}. "
                "Run the full pipeline first: uv run python -m src.run_pipeline"
            )
        embeddings_path = self.output_path / "topic_embeddings.json"
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"topic_embeddings.json not found at {embeddings_path}. "
                "Run the full pipeline first: uv run python -m src.run_pipeline"
            )

        logger.info("Loading topic model…")
        modeler = TopicModeler(output_path=str(self.output_path))
        topic_model = modeler.load_topic_model()

        logger.info("Loading topic embeddings…")
        with open(embeddings_path) as f:
            topic_embeddings: dict[str, list[float]] = json.load(f)

        neighbor_ids = self._find_neighbor_topic_ids(topic_embeddings)
        logger.info(f"Neighbor topic IDs selected: {neighbor_ids}")

        posts: list[Post] = []
        for neighbor_id in neighbor_ids:
            topic_words = topic_model.get_topic(neighbor_id)
            if not topic_words:
                logger.warning(f"No keywords found for topic {neighbor_id}; skipping.")
                continue
            keywords = [word for word, _ in topic_words]
            for post_idx in range(len(posts), len(posts) + self.args.posts_per_neighbor):
                text = self._generate_post_text(keywords)
                posts.append(self._make_post(text, post_idx))

        logger.info(f"Generated {len(posts)} noise posts.")
        NOISE_OUTPUT_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
        with open(NOISE_OUTPUT_FILEPATH, "w") as f:
            json.dump([post.model_dump(mode="json") for post in posts], f, indent=2)
        logger.info(f"Saved noise posts → {NOISE_OUTPUT_FILEPATH}")

    def cleanup_noise(self) -> None:
        """Delete all noise posts from Elasticsearch and remove the JSON file."""
        try:
            client = get_es_client()
            client.info()
            deletion_result = client.delete_by_query(
                index=INDEX_NAME,
                body={"query": {"term": {"post_author": "noise"}}},
            )
            logger.info(f"Deleted {deletion_result['deleted']} noise posts from Elasticsearch.")
        except Exception as e:
            logger.warning(f"Elasticsearch cleanup skipped: {e}")

        if NOISE_OUTPUT_FILEPATH.exists():
            NOISE_OUTPUT_FILEPATH.unlink()
            logger.info(f"Deleted {NOISE_OUTPUT_FILEPATH}.")


if __name__ == "__main__":
    from src.run_pipeline import pipeline as run_full_pipeline

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    # Step 1 — generate and save noise posts to output/noise_posts.json
    NoiseGenerator(NoiseGeneratorArgs()).run()

    # Step 2 — re-run the full pipeline (preprocess sample + noise, retrain model)
    run_full_pipeline()
