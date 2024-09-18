import asyncio
import json
import logging
from typing import List

from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer

from index import INDEX_NAME, create_index
from models import PostDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load sample posts from the JSON file
def load_sample_posts(file_path: str) -> List[dict]:
    with open(file_path, "r") as file:
        posts = json.load(file)
    logger.info(f"Loaded {len(posts)} posts from {file_path}.")
    return posts


# Convert loaded posts to the Pydantic model
def convert_to_pydantic(posts_data: List[dict]) -> List[PostDocument]:
    return [PostDocument(**post) for post in posts_data]


# Create embeddings for posts  using SentenceTransformers' built-in batching
async def create_embeddings_for_posts(posts: List[PostDocument]):
    logger.debug("Creating embeddings for posts.")
    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
    )  # Load a pre-trained model

    # Extract all post texts (pre-process them)
    texts = [post.preprocess_text() for post in posts]

    # Encode all texts
    # Typically this would be run for each post individually upon post creation,
    # but for demonstration purposes...
    embeddings = await asyncio.to_thread(
        model.encode,
        texts,
        show_progress_bar=False,
        clean_up_tokenization_spaces=True,
    )

    # Assign embeddings back to the respective posts
    for post, embedding in zip(posts, embeddings):
        post.doc_embedding = embedding.tolist()

    logger.info("Completed creating embeddings for posts.")


# Store posts in OpenSearch using bulk upload
def store_posts_in_opensearch(
    opensearch_client, posts: List[PostDocument], batch_size=50
):
    logger.debug(f"Storing {len(posts)} posts in OpenSearch.")

    # Prepare the actions for bulk upload
    actions = [
        {
            "_index": INDEX_NAME,
            "_id": post.post_id,
            "_source": post.model_dump(),
        }
        for post in posts
    ]

    # Execute the bulk upload
    success, failed = helpers.bulk(opensearch_client, actions, chunk_size=batch_size)
    logger.info(f"Successfully stored {success} posts in OpenSearch.")
    if failed:
        logger.error(f"Failed to store {failed} posts in OpenSearch.")


# Main function to load, process, and store posts
async def main():
    # Initialize OpenSearch client
    opensearch_client = OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}], http_compress=True
    )

    # Create the index
    create_index(opensearch_client)

    # Load sample posts from the JSON file
    sample_posts = load_sample_posts("sample_posts.json")

    # Convert posts to Pydantic models
    posts = convert_to_pydantic(sample_posts)

    # Pre-process text and create embeddings for each post
    await create_embeddings_for_posts(posts)

    # Store the posts in OpenSearch using batch upload
    store_posts_in_opensearch(opensearch_client, posts)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
