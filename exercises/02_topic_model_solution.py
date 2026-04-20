"""Exercise 2 Solution: Topic Model Pipeline"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUTPUT_PATH = Path(__file__).parent.parent / "output"
DOC_INDEX_PATH = OUTPUT_PATH / "doc_index.json"
MODEL_PATH = OUTPUT_PATH / "bertopic_model"
ASSIGNMENTS_PATH = OUTPUT_PATH / "topic_assignments.csv"
LOCALIZED_EMBEDDINGS_PATH = OUTPUT_PATH / "topic_embeddings_localized.json"


def load_posts() -> list[dict]:
    with open(DOC_INDEX_PATH) as f:
        data = json.load(f)
    return list(data.values())


def load_topic_model() -> BERTopic:
    return BERTopic.load(str(MODEL_PATH))


def print_topic_summary(topic_model: BERTopic):
    topic_info = topic_model.get_topic_info()
    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            continue
        keywords = [w for w, _ in topic_model.get_topic(topic_id)[:5]]
        print(f"  Topic {topic_id:2d} ({row['Count']:3d} docs): {', '.join(keywords)}")


def save_localized_embeddings(topic_model: BERTopic, embedding_model: SentenceTransformer):
    topic_info = topic_model.get_topic_info()
    topic_ids = [t for t in topic_info["Topic"].tolist() if t != -1]

    localized = {}
    for topic_id in topic_ids:
        keywords = [w for w, _ in topic_model.get_topic(topic_id)]
        embedding = embedding_model.encode(" ".join(keywords), convert_to_numpy=True)
        localized[topic_id] = embedding.tolist()

    with open(LOCALIZED_EMBEDDINGS_PATH, "w") as f:
        json.dump(localized, f)

    print(f"Saved localized embeddings for {len(localized)} topics → {LOCALIZED_EMBEDDINGS_PATH}")
    return localized


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
