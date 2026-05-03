"""
====================
PREPROCESS POST PIPELINE
====================
Preprocessing pipeline: caption images, generate embeddings, store posts.

Run:
    uv run python -m src.preprocess

Steps:
  1. Load posts from sample_posts.json
  2. Caption any image-only posts that lack a caption (BLIP vision model)
  3. Batch-embed all posts (combines post_text + image_caption when both exist)
  4. Save to Elasticsearch (if running) and to output/processed_posts.json


"""

import json
import logging
from pathlib import Path

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.config import (
    ELASTICSEARCH_URL,
    EMBEDDING_MODEL_NAME,
    OUTPUT,
    REPO,
    VISION_MODEL_NAME,
)
from src.data_models import PostDocument

logger = logging.getLogger(__name__)

INPUT_FILEPATH = REPO / "sample_posts.json"
OUTPUT_FILEPATH = OUTPUT / "processed_posts.json"


def extract_embedding_text(postdoc: PostDocument) -> str:
    """
    Build the text string passed to the embedding model.

    Combines the elements of a PostDocument that should be included in the
    embedding into a single string. Elements that should not be included
    (e.g. post_id, image_url) are ignored.
    """
    # Extract the text elements to embed
    text = postdoc.preprocess_text().strip()
    # Check if the post has an image caption (i.e. image converted to text)
    caption = (postdoc.image_caption or "").strip()
    if text and caption:
        return f"{text} {caption}"
    return text or caption


class PreprocessingPipeline:
    """End-to-end preprocessing pipeline: load → caption → embed → store."""

    def __init__(
        self,
        input_filepath: Path = INPUT_FILEPATH,
        output_filepath: Path = OUTPUT_FILEPATH,
    ) -> None:
        """Load models and attempt to connect to Elasticsearch."""
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.elasticsearch_client = self._connect_elasticsearch()
        logger.info(f"Loading embedding model {EMBEDDING_MODEL_NAME}…")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded.")
        logger.info(f"Loading vision model {VISION_MODEL_NAME}…")
        self._vision_processor, self._vision_model = self._load_vision_model()
        logger.info("Vision model loaded.")

    def _load_vision_model(self) -> tuple:
        """Load vision model processor and model for image captioning."""
        processor = BlipProcessor.from_pretrained(
            VISION_MODEL_NAME,
            use_fast=False,  # fast image processor incompatible so we disable it.
            use_fast_tokenizer=True,
        )
        model = BlipForConditionalGeneration.from_pretrained(
            VISION_MODEL_NAME,
            use_safetensors=True,
            # force_download=True, # uncomment to clear model cache
        )
        return processor, model.to("cpu")

    def _connect_elasticsearch(self) -> Elasticsearch | None:
        """
        Return a connected Elasticsearch client, or None if Elasticsearch is
        unavailable.
        """
        try:
            client = Elasticsearch(ELASTICSEARCH_URL)
            client.info()
            return client
        except Exception:
            return None

    def load_postdocs(self) -> list[PostDocument]:
        """Load raw posts from sample_posts.json and parse into PostDocument objects."""
        logger.info(f"Loading {self.input_filepath}…")
        with open(self.input_filepath) as f:
            return [PostDocument(**p) for p in json.load(f)]

    def _caption_single_post(self, postdoc: PostDocument) -> None:
        """
        Run vision model on one postdoc and set image_caption in-place.
        For most projects the HuggingFace pipeline API is simpler:
            from transformers import pipeline
            captioner = pipeline("image-to-text",
                                 model="Salesforce/blip-image-captioning-base",
                                 model_kwargs={"use_safetensors": True})
            caption = captioner(image, max_new_tokens=125)[0]["generated_text"]
        To illustrate the underlying steps, and to ensure cross-device
        compatibility, we use the processor and model directly here.
        """
        if not postdoc.image_url:
            return

        img_path = REPO / postdoc.image_url
        if not img_path.exists():
            logger.warning(f"Image file not found: {img_path}")
            return
        image = Image.open(img_path).convert("RGB")
        logger.info(f"Captioning {img_path.name} (size: {image.size})…")

        ### EXERCISE ###
        # Use self._vision_processor to prepare the image as a tensor
        # Use self._vision_model.generate() to produce caption token IDs
        # Decode the token IDs back to a string with self._vision_processor.decode()
        # Store the result: postdoc.image_caption = caption

    def caption_images(self, postdocs: list[PostDocument]) -> list[PostDocument]:
        """Caption image-only posts"""
        needs_caption = [postdoc for postdoc in postdocs if postdoc.image_url]
        if not needs_caption:
            logger.info("No image posts need captioning.")
            return postdocs
        for postdoc in needs_caption:
            self._caption_single_post(postdoc)
        return postdocs

    def generate_embeddings(self, postdocs: list[PostDocument]) -> list[PostDocument]:
        """
        Embed all postdocs using the embedding model.
        """
        logger.info(f"Embedding {len(postdocs)} postdocs…")
        ### EXERCISE ###
        # For each postdoc, extract the text to embed (hint: use the
        # extract_embedding_text function above)
        # Use the embedding model to generate an embedding vector for the text
        # Store the embedding vector on the postdoc (postdoc.doc_embedding = embedding)
        return postdocs

    def save_to_elasticsearch(self, postdocs: list[PostDocument]) -> None:
        """Index postdocs into Elasticsearch, skipping any already present."""
        from src.es_index import INDEX_NAME, create_index

        client = self.elasticsearch_client
        if client is None:
            return

        create_index(client)
        new_postdocs = [
            postdoc
            for postdoc in postdocs
            if not client.exists(index=INDEX_NAME, id=postdoc.post_id)
        ]
        for postdoc in new_postdocs:
            client.index(index=INDEX_NAME, id=postdoc.post_id, body=postdoc.model_dump(mode="json"))
        logger.info(
            f"Stored {len(new_postdocs)} new postdocs in Elasticsearch ({len(postdocs) - len(new_postdocs)} already present)."
        )

    def save_processed_posts(self, postdocs: list[PostDocument]) -> None:
        """Write processed posts to output_filepath, keyed by post_id.

        Merges with any existing file, skipping post IDs already present.
        """
        self.output_filepath.parent.mkdir(exist_ok=True)
        existing: dict = {}
        if self.output_filepath.exists():
            with open(self.output_filepath) as f:
                existing = json.load(f)
        new_docs = {
            postdoc.post_id: postdoc.model_dump(mode="json")
            for postdoc in postdocs
            if postdoc.post_id not in existing
        }
        existing.update(new_docs)
        with open(self.output_filepath, "w") as f:
            json.dump(existing, f)
        logger.info(
            f"Saved {self.output_filepath.name} ({len(existing)} postdocs total, {len(new_docs)} new)."
        )

    def run(self) -> None:
        """Run the full preprocessing pipeline: load → caption → embed → store."""
        postdocs = self.load_postdocs()
        postdocs = self.caption_images(postdocs)
        postdocs = self.generate_embeddings(postdocs)

        if self.elasticsearch_client:
            logger.info("Elasticsearch available — saving postdocs.")
            self.save_to_elasticsearch(postdocs)
        else:
            logger.info("Elasticsearch not available — using disk only.")

        self.save_processed_posts(postdocs)
        logger.info("Preprocessing complete. Run: uv run python -m src.topic_model")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    PreprocessingPipeline().run()
