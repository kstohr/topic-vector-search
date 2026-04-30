"""
Evaluation metrics for topic search:

- Evaluation-k (e.g., K=100): number of results returned by search.
This is passed to the "size" parameter of the search query and determines the
total number of results returned by search for evaluation. This should be large
enough to capture all relevant posts for fair evaluation, but not so large that
it includes a lot of noise. Should be tuned based on corpus size and typical
search depth in your application.

- Recall@K: coverage of topic within top-K results (e.g., K=100)
How many of the assigned posts are retrieved by search. Should be large enough
to capture all relevant posts for fair evaluation. Here we set it to a value
greater than the number of posts assigned to most topics.

- Precision@K: quality of top-K results (e.g., K=8)
How many of the assigned posts appear in the top K results; where "k" is set to
a typical value for search results displayed before pagination (e.g., 8 or 10).

- Baseline precision: expected precision of a random baseline for topic of this
size within the dataset
Random baseline precision: If we were to randomly select K posts from the
entire corpus, what precision would we expect? This can be computed as (number
of relevant posts) / (total number of posts). This baseline helps us
understand how much better our search is compared to random chance. We compute
this for each topic and include it in the evaluation metrics for context.
"""

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, PositiveInt

from src.search import TopicSearchArgs, run_search_by_topic

DEFAULT_EVAL_K = 100
DEFAULT_RECALL_K = 100
DEFAULT_PRECISION_K = 8


def compute_random_baseline(topic_size: int, dataset_size: int):
    """Computes the expected precision of a random baseline for a given topic."""
    return topic_size / dataset_size


class TopicSearchMetricArgs(BaseModel):
    """Input arguments for compute_topic_search_eval_metrics."""

    dataset_size: PositiveInt  # Used to compute random baseline precision
    retrieved_ids: list[str]
    topic_post_ids: set[str]
    recall_k: PositiveInt = DEFAULT_RECALL_K
    precision_k: PositiveInt = DEFAULT_PRECISION_K


class TopicSearchMetrics(BaseModel):
    """Precision, recall, and baseline metrics for a single topic search evaluation."""

    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    baseline: float = 0.0
    num_posts_assigned_to_topic: int = 0
    num_retrieved_by_search: int = 0
    precision_hits: int = 0
    recall_hits: int = 0


def compute_topic_search_eval_metrics(args: TopicSearchMetricArgs) -> TopicSearchMetrics:
    """
    Computes:
    - Recall@K: coverage of topic within top-K results (e.g., K=20)
    - Precision@K: quality of top-K results (e.g., K=8)
    - Baseline precision: expected precision of a random baseline for topic of
    this size within the corpus

    Assumes retrieved_ids are ordered by relevance (highest score first).
    """
    # --- Precision@K ---
    top_precision_k = args.retrieved_ids[: args.precision_k]
    precision_hits = sum(pid in args.topic_post_ids for pid in top_precision_k)
    precision_at_k = precision_hits / args.precision_k

    # --- Recall@K ---
    top_recall_k = args.retrieved_ids[: args.recall_k]
    recall_hits = sum(pid in args.topic_post_ids for pid in top_recall_k)
    recall_at_k = recall_hits / len(args.topic_post_ids)

    # --- Baseline precision ---
    baseline = compute_random_baseline(len(args.topic_post_ids), args.dataset_size)

    return TopicSearchMetrics(
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        baseline=baseline,
        num_posts_assigned_to_topic=len(args.topic_post_ids),
        num_retrieved_by_search=len(args.retrieved_ids),
        precision_hits=precision_hits,
        recall_hits=recall_hits,
    )


class TopicEvalArgs(BaseModel):
    """Input arguments for evaluate_topics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    corpus_size: int
    labels: dict[int, dict]
    assignments: pd.DataFrame
    topic_embeddings: dict[int, list[float]]
    searcher: Any  # InMemorySemanticSearcher | SemanticSearcher
    keywords: dict[int, list[str]] | None = None
    eval_k: int = DEFAULT_EVAL_K


class TopicEvalRow(TopicSearchMetrics):
    """A single row in the topic evaluation DataFrame."""

    topic: str
    keywords: str
    assigned_posts: int
    avg_search_score: float


def evaluate_topics(args: TopicEvalArgs) -> pd.DataFrame:
    """
    For each topic, run a semantic search using its embedding and compute
    precision metrics against the ground-truth topic assignments.

    Note: Search depth can change the results significantly.

    Columns returned:
      topic           — label with emoji
      keywords        — top keywords by c-TF-IDF score, computed during BERTopic training
      assigned_posts  — how many posts were assigned to this topic by BERTopic
      avg_search_score — average semantic search score for the evaluation results
      ** Topic search evaluation metrics (precision@K, recall@K, random baseline
      precision, etc.) computed by compute_topic_search_eval_metrics

    """
    assignments = args.assignments
    labels = args.labels
    topic_embeddings = args.topic_embeddings
    searcher = args.searcher
    keywords = args.keywords

    assigned_by_topic = {
        tid: set(assignments.loc[assignments["topic_id"] == tid, "post_id"]) for tid in labels
    }

    rows = []
    for topic_id, info in sorted(labels.items()):
        emb = np.array(topic_embeddings[topic_id], dtype=np.float32)
        assigned = assigned_by_topic.get(topic_id, set())
        kws = keywords.get(topic_id, [])[:10] if keywords else []

        # Next run search for evaluation
        search_results = run_search_by_topic(TopicSearchArgs(embedding=emb, searcher=searcher, top_k=args.eval_k))

        # Display general scores
        scores = [r["score"] for r in search_results]

        eval: TopicSearchMetrics = compute_topic_search_eval_metrics(
            args=TopicSearchMetricArgs(
                dataset_size=args.corpus_size,
                retrieved_ids=[r.get("post_id", "") for r in search_results],
                topic_post_ids=assigned,
                recall_k=DEFAULT_RECALL_K,
                precision_k=DEFAULT_PRECISION_K,
            )
        )

        # define the row for this topic
        row = TopicEvalRow(
            topic=info["label"],
            keywords=", ".join(kws),
            assigned_posts=len(assigned),
            avg_search_score=np.mean(scores) if scores else 0.0,
            **eval.model_dump(),
        )
        rows.append(row.model_dump())

    cols = [
        "topic",
        "keywords",
        "assigned_posts",
        "avg_search_score",
        *TopicSearchMetrics.model_fields.keys(),
    ]
    df = pd.DataFrame(rows)[cols]  # Ensure consistent column order
    return df


class SearchResultRowsArgs(BaseModel):
    """Input arguments for build_search_result_rows."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    results: list[dict]
    posts_by_id: dict[str, dict]
    assignments: pd.DataFrame
    labels: dict[int, dict]


class SearchResultRow(BaseModel):
    """A single search result row enriched with topic label and display text."""

    score: float
    post: str
    topic: str
    image_url: str
    post_id: str


# Helper to join search results with topic assignments and post metadata for
# display in the demo app. Not strictly part of evaluation, but useful for understanding results.
def build_search_result_rows(args: SearchResultRowsArgs) -> list[SearchResultRow]:
    """
    Join search results with topic assignments and post metadata.
    Returns a list of SearchResultRow ready for display or ranking.
    """
    id_to_label = {tid: v["label"] for tid, v in args.labels.items()}

    rows = []
    for r in args.results:
        pid = r.get("post_id", "")
        post = args.posts_by_id.get(pid, {})
        text = r.get("post_text", "").strip()
        caption = (post.get("image_caption") or "").strip()

        if text:
            display_text = text
        elif caption:
            display_text = f"[image] {caption}"
        else:
            display_text = "[image — no caption yet]"

        topic_label = ""
        match = args.assignments.loc[args.assignments["post_id"] == pid, "topic_id"]
        if len(match):
            tid = int(match.values[0])
            topic_label = id_to_label.get(tid, f"Topic {tid}")

        rows.append(SearchResultRow(
            score=round(r.get("score", 0), 3),
            post=display_text,
            topic=topic_label,
            image_url=post.get("image_url", ""),
            post_id=pid,
        ))

    return rows
