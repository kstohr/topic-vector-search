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

-------------------------
Embedding explanation:
Text → tokens

"The cat sat" → [1996, 4937, 2938]

Token → vector (lookup)

1996 → embedding_matrix[1996] → vector of floats

Those vectors are passed through layers
Each layer applies:

linear transformations (matrix multiplies)
attention (mixing information across tokens)
nonlinearities

So each token’s vector is updated based on other tokens

Final token vectors exist
At this point, each token has a contextualized vector
Pooling
Those vectors are combined (e.g., mean) → one document vector
-------------------------
Image model tensor explanation:
What it starts as (raw pixel)

A pixel in an image is typically:

R = 120, G = 200, B = 30   (values from 0–255)
Step 1: scale to 0–1
R = 120 / 255 ≈ 0.47
G = 200 / 255 ≈ 0.78
B = 30  / 255 ≈ 0.12
Step 2: normalize (center + scale)

Models expect values centered around 0:

value = (value - mean) / std

So you might end up with:

R ≈ -0.1
G ≈  1.2
B ≈ -1.5
What the float represents

Each float is:

"How bright this pixel is in this color channel, relative to what the model expects"

"""

import json
import logging

from elasticsearch import Elasticsearch
from PIL import Image
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

    def __init__(self) -> None:
        """Load models and attempt to connect to Elasticsearch."""
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
        logger.info("Loading sample_posts.json…")
        with open(REPO / "sample_posts.json") as f:
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

        # Load the image from disk and convert to RGB (BLIP expects 3-channel input)
        img_path = REPO / postdoc.image_url
        if not img_path.exists():
            logger.warning(f"Image file not found: {img_path}")
            return
        logger.info(f"Captioning {img_path.name}…")
        image = Image.open(img_path).convert("RGB")  # BLIP expects 3-channel RGB input

        # Process the image
        # Image is resized and normalized according to model requirements.
        # The output is a tensor (4D array, shape [1, 3, H, W])
        # - batch size = 1 (number of images, we could do more at once)
        # - 3 color channels (RGB)
        # - height = 224 (# pixels in vertical dimension)
        # - width = 224 (# pixels in horizontal dimension)
        # This means:
        #     1 image (leading dimension) with
        #     3 layers (R, G, B)
        #     each layer is a 224×224 grid of values derived from the original
        #     pixels
        #
        inputs = self._vision_processor(images=image, return_tensors="pt").to("cpu")

        # Pre-trained vision model (transformer-based) learns patterns in images
        # It generates a sequence of text tokens describing the image
        # The input is the processed image tensor
        # The output is a sequence of token (word) IDs, which we decode back to text
        output = self._vision_model.generate(**inputs, max_new_tokens=256)
        caption = self._vision_processor.decode(output[0], skip_special_tokens=True)
        postdoc.image_caption = caption
        logger.info(f"  → {caption}")

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
        # Extract the text to embed for each postdoc.
        # Builds the text string that is passed to the model.
        # Combines elements of the structured document into a single string,
        # Excludes elements that should not be included (e.g. author, post_id,
        # image_url).
        # Includes enriched text (i.e. emojis converted to text, image captions,
        # any other post attributes that would be useful to search.)
        # In this case we embed the document as a single string, but you could also embed
        # sentences or other chunks of text depending on your use case.
        texts = [extract_embedding_text(postdoc) for postdoc in postdocs]

        # The embedding model is trained on a corpus of documents. Each word in each
        # document is positioned in the "embedding space" based on the contexts
        # it appears in across the corpus. Words that frequently appear in
        # similar contexts will be positioned closer together in the embedding
        # space.

        # When we pass a new document to the model, it generates a vector (list
        # of
        # floats) that indicates where in the embedding space that document
        # is positioned.

        # The output is a vector (list of floats) that has length equal to the
        # embedding dimension of the model (e.g. 384 for the all-MiniLM model).

        # Similar documents will have similar vectors. So, if two posts talk
        # about similar topics, they will have similar embeddings, and will be
        # positioned closer together in the embedding space. This allows us to
        # perform semantic search and topic modeling based on the content of
        # the posts.

        # Batch embed texts.
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        # Store the embedding vector on the PostDocument
        for postdoc, embedding in zip(postdocs, embeddings, strict=True):
            # Convert the Numpy array to a list so it can be JSON-serialized and stored to
            # Elasticsearch. When we load it back for modeling, we'll convert it back to an array.
            postdoc.doc_embedding = embedding.tolist()
        return postdocs

    def save_to_elasticsearch(self, postdocs: list[PostDocument]) -> None:
        """Index all postdocs into Elasticsearch."""
        from src.es_index import INDEX_NAME, create_index

        client = self.elasticsearch_client
        if client is None:
            return

        create_index(client)
        for postdoc in postdocs:
            client.index(index=INDEX_NAME, id=postdoc.post_id, body=postdoc.model_dump(mode="json"))
        logger.info(f"Stored {len(postdocs)} postdocs in Elasticsearch.")

    def save_processed_posts(self, postdocs: list[PostDocument]) -> None:
        """Write output/processed_posts.json keyed by post_id for downstream steps."""
        OUTPUT.mkdir(exist_ok=True)
        doc_index = {postdoc.post_id: postdoc.model_dump(mode="json") for postdoc in postdocs}
        with open(OUTPUT / "processed_posts.json", "w") as f:
            json.dump(doc_index, f)
        logger.info(f"Saved processed_posts.json ({len(doc_index)} postdocs).")

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
