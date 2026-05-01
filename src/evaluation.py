"""
====================
TOPIC SEARCH EVALUATION METRICS
====================

Below are the key metrics we compute to evaluate the quality of our topic search results:

- Evaluation-k (e.g., k=100; *retrieval depth*)
How many results to retrieve for evaluation. This is passed to the "size"
parameter of the search query and determines the total number of results
returned by search for evaluation. This should be large enough to capture all
relevant posts for fair evaluation, but not so large that it includes a lot of
noise. Should be tuned based on corpus size and typical search depth in your application.

- Recall@k: coverage of topic within top-K results (e.g., k=100; **evaluation
  cutoff**)
How many of the posts assigned to this topic appear in the top K results; where
"k" is set to a value that captures the typical depth returned by search
results. Should be large enough to capture all relevant posts for fair
evaluation. Here we set it to a value greater than the number of posts assigned to a typical topic.

- Precision@k: quality of top-K results (e.g., k=8, **display cutoff**)
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

import logging
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, PositiveInt

from src.search import TopicSearchArgs, run_search_by_topic

DEFAULT_EVAL_K = 100  # Retrieval depth
DEFAULT_RECALL_K = 100  # Evaluation cutoff
DEFAULT_PRECISION_K = 8  # Display cutoff


logger = logging.getLogger(__name__)


class ComputePrecisionResp(BaseModel):
    """Response from compute_precision_at_k."""

    precision: float
    hits: int


def compute_precision_at_k(
    retrieved_ids: list[str],
    topic_post_ids: set[str],
    k: int,
) -> ComputePrecisionResp:
    """Computes Precision@K: the proportion of the top-K retrieved posts that
    are relevant to the topic.
    """
    logger.info(f"Computing Precision@{k}")
    # Retrieval may return < K results. This may be less than K
    top_k_ids = retrieved_ids[:k]
    if not top_k_ids:
        return ComputePrecisionResp(precision=0.0, hits=0)
    hits = sum(pid in topic_post_ids for pid in top_k_ids)
    return ComputePrecisionResp(
        precision=hits / len(top_k_ids),
        hits=hits,
    )


class ComputeRecallResp(BaseModel):
    """Response from compute_recall_at_k."""

    recall: float
    hits: int


def compute_recall_at_k(
    retrieved_ids: list[str],
    topic_post_ids: set[str],
    k: int,
) -> ComputeRecallResp:
    """
    Computes Recall@K: the proportion of relevant posts that are retrieved in
    the top-K results.
    """
    logger.info(f"Computing Recall@{k}")
    top_k_ids = retrieved_ids[:k]
    if not topic_post_ids:
        return ComputeRecallResp(recall=0.0, hits=0)
    hits = sum(pid in topic_post_ids for pid in top_k_ids)
    return ComputeRecallResp(
        recall=hits / len(topic_post_ids),
        hits=hits,
    )


def compute_random_baseline(
    topic_size: int,
    dataset_size: int,
) -> float:
    """
    Computes the expected precision of a random baseline for a given topic.
    """
    logger.info("Computing Baseline Precision")
    if dataset_size == 0:
        return 0.0  # Avoid division by zero.
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

    # --- Baseline precision ---
    baseline = compute_random_baseline(
        len(args.topic_post_ids),
        args.dataset_size,
    )

    # --- Precision@K ---
    precision_at_k = compute_precision_at_k(
        args.retrieved_ids,
        args.topic_post_ids,
        args.precision_k,
    )

    # --- Recall@K ---
    recall_at_k = compute_recall_at_k(
        args.retrieved_ids,
        args.topic_post_ids,
        args.recall_k,
    )

    return TopicSearchMetrics(
        precision_at_k=precision_at_k.precision,
        recall_at_k=recall_at_k.recall,
        baseline=baseline,
        num_posts_assigned_to_topic=len(args.topic_post_ids),
        num_retrieved_by_search=len(args.retrieved_ids),
        precision_hits=precision_at_k.hits,
        recall_hits=recall_at_k.hits,
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
        topic_id: set(assignments.loc[assignments["topic_id"] == topic_id, "post_id"])
        for topic_id in labels
    }

    rows = []
    for topic_id, info in sorted(labels.items()):
        topic_embedding = np.array(topic_embeddings[topic_id], dtype=np.float32)
        assigned = assigned_by_topic.get(topic_id, set())
        topic_keywords = keywords.get(topic_id, [])[:10] if keywords else []

        # Next run search for evaluation
        search_results = run_search_by_topic(
            TopicSearchArgs(embedding=topic_embedding, searcher=searcher, top_k=args.eval_k)
        )

        # Display general scores
        scores = [result["score"] for result in search_results]

        metrics: TopicSearchMetrics = compute_topic_search_eval_metrics(
            args=TopicSearchMetricArgs(
                dataset_size=args.corpus_size,
                retrieved_ids=[result.get("post_id", "") for result in search_results],
                topic_post_ids=assigned,
                recall_k=DEFAULT_RECALL_K,
                precision_k=DEFAULT_PRECISION_K,
            )
        )

        # define the row for this topic
        row = TopicEvalRow(
            topic=info["label"],
            keywords=", ".join(topic_keywords),
            assigned_posts=len(assigned),
            avg_search_score=np.mean(scores) if scores else 0.0,
            **metrics.model_dump(),
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
    Returns a list of SearchResultRow ready for display in the demo app.
    """
    id_to_label = {tid: v["label"] for tid, v in args.labels.items()}

    rows = []
    for result in args.results:
        post_id = result.get("post_id", "")
        post = args.posts_by_id.get(post_id, {})
        text = result.get("post_text", "").strip()
        caption = (post.get("image_caption") or "").strip()

        if text:
            display_text = text
        elif caption:
            display_text = f"[image] {caption}"
        else:
            display_text = "[image — no caption yet]"

        topic_label = ""
        match = args.assignments.loc[args.assignments["post_id"] == post_id, "topic_id"]
        if len(match):
            topic_id = int(match.values[0])
            topic_label = id_to_label.get(topic_id, f"Topic {topic_id}")

        rows.append(
            SearchResultRow(
                score=round(result.get("score", 0), 3),
                post=display_text,
                topic=topic_label,
                image_url=post.get("image_url", ""),
                post_id=post_id,
            )
        )

    return rows
