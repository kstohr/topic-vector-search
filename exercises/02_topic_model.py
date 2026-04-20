"""
Exercise 2: Topic Model Pipeline
=================================
Complete the TODOs below to build the topic model pipeline.

Run with:
    python exercises/02_topic_model.py

After completing this exercise, you will have:
- A loaded/trained BERTopic model
- topic_assignments.csv saved to output/
- Localized keyword embeddings saved to output/topic_embeddings_localized.json
  (needed for Exercise 3 Part B)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUTPUT_PATH = Path(__file__).parent.parent / "output"
DOC_INDEX_PATH = OUTPUT_PATH / "doc_index.json"
MODEL_PATH = OUTPUT_PATH / "bertopic_model"
ASSIGNMENTS_PATH = OUTPUT_PATH / "topic_assignments.csv"
LOCALIZED_EMBEDDINGS_PATH = OUTPUT_PATH / "topic_embeddings_localized.json"


# ---------------------------------------------------------------------------
# Step 1: Load posts and their embeddings from the saved doc index
# ---------------------------------------------------------------------------

def load_posts() -> list[dict]:
    """Load posts from output/doc_index.json."""
    # TODO: open DOC_INDEX_PATH, parse JSON, return a list of post dicts
    # Hint: json.loads / json.load — the file is a dict keyed by post_id;
    #       return list(data.values())
    raise NotImplementedError("Complete Step 1")


# ---------------------------------------------------------------------------
# Step 2: Load the pre-trained BERTopic model
# ---------------------------------------------------------------------------

def load_topic_model() -> BERTopic:
    """Load BERTopic model from output/bertopic_model/."""
    # TODO: use BERTopic.load(str(MODEL_PATH)) and return the model
    raise NotImplementedError("Complete Step 2")


# ---------------------------------------------------------------------------
# Step 3: Print a topic summary
# ---------------------------------------------------------------------------

def print_topic_summary(topic_model: BERTopic):
    """Print topic IDs, counts, and top 5 keywords for each topic."""
    # TODO: call topic_model.get_topic_info() to get a DataFrame,
    #       then iterate over rows and print each topic's ID and top keywords.
    # Hint: topic_model.get_topic(topic_id) returns [(word, score), ...]
    raise NotImplementedError("Complete Step 3")


# ---------------------------------------------------------------------------
# Step 4: Compute and save localized keyword embeddings
# ---------------------------------------------------------------------------

def save_localized_embeddings(topic_model: BERTopic, embedding_model: SentenceTransformer):
    """
    For each topic, encode its top keywords as a single embedding and save
    the results to output/topic_embeddings_localized.json.

    This is the "localized" embedding strategy: instead of using BERTopic's
    internal centroid (which captures the full cluster geometry), we encode
    only the top keywords — a focused, query-ready embedding.
    """
    topic_info = topic_model.get_topic_info()
    # Exclude the outlier topic (-1)
    topic_ids = [t for t in topic_info["Topic"].tolist() if t != -1]

    localized = {}
    for topic_id in topic_ids:
        # TODO: get the top keywords for this topic
        # Hint: topic_model.get_topic(topic_id) → [(word, score), ...]
        keywords = None  # replace with your code

        # TODO: join keywords into a single string and encode with embedding_model
        # Hint: embedding_model.encode("word1 word2 word3", convert_to_numpy=True)
        embedding = None  # replace with your code

        localized[topic_id] = embedding.tolist()

    with open(LOCALIZED_EMBEDDINGS_PATH, "w") as f:
        json.dump(localized, f)

    print(f"Saved localized embeddings for {len(localized)} topics → {LOCALIZED_EMBEDDINGS_PATH}")
    return localized


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("=== Step 1: Loading posts ===")
    posts = load_posts()
    print(f"Loaded {len(posts)} posts")

    print("\n=== Step 2: Loading topic model ===")
    topic_model = load_topic_model()
    print("Model loaded")

    print("\n=== Step 3: Topic summary ===")
    print_topic_summary(topic_model)

    print("\n=== Step 4: Saving localized embeddings ===")
    save_localized_embeddings(topic_model, embedding_model)

    print("\nDone! Now open notebooks/03_search_evaluation.ipynb to evaluate.")
