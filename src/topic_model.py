"""
====================
TOPIC MODEL TRAINING
====================

Topic model training pipeline.

Run:
    uv run python -m src.topic_model

Reads post documents from Elasticsearch (if running) or output/processed_posts.json,
trains BERTopic, and writes all output artifacts:
  output/bertopic_model/
  output/topic_assignments.csv
  output/topic_labels.json
  output/topic_keyword_embeddings.json
  output/topics.json
  output/probabilities.json
  output/topic_visualization.html  (and other BERTopic charts)

LLM labeling priority:
  1. Ollama at localhost:11434  (no API key needed)
  2. OPENAI_API_KEY env var     (OpenAI API)
  3. KeyBERT keywords only      (no LLM)
"""

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.representation import OpenAI as BertTopicOpenAI
from bertopic.vectorizers import ClassTfidfTransformer
from elasticsearch import Elasticsearch
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from src.ai_labeler import build_llm_representation
from src.config import (
    ELASTICSEARCH_URL,
)
from src.config import (
    EMBEDDING_MODEL_NAME as EMBEDDING_MODEL,
)
from src.data_models import PostDocument
from src.preprocess import extract_embedding_text
from src.retrieve_postdocs import retrieve_postdocs_from_disk, retrieve_postdocs_from_elasticsearch

logger = logging.getLogger(__name__)

RANDOM_SEED = 99

LABEL_PROMPT = """
I have a topic that contains the following documents:
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the information above, extract a short but highly descriptive topic
label of at most 3 words. Make sure it is in the following format:
topic: <topic label>
"""

TOP_N_SEARCH_EMBEDDING_KEYWORDS = 3


class NoTopicsFoundError(Exception):
    """Raised when BERTopic finds no topics. Try adjusting model parameters."""


class ProccessedPostsNotFoundError(FileNotFoundError):
    """Raised when processed_posts.json is missing for topic model training."""


class TopicModeler:
    """Orchestrates BERTopic training, artifact storage, and visualization."""

    def __init__(self, index_name: str = "post_docs", output_path: str = "output"):
        """Load the embedding model and attempt to connect to Elasticsearch."""
        self.index_name = index_name
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.topic_model: BERTopic | None = None
        self.doc_index: dict[str, PostDocument] = {}
        self.elasticsearch_client = self._connect_elasticsearch()
        self.n_topics = 0  # Number of topics found (excluding outliers)

    def _connect_elasticsearch(self) -> Elasticsearch | None:
        """Return a connected Elasticsearch client, or None if unavailable."""
        try:
            client = Elasticsearch(ELASTICSEARCH_URL)
            client.info()
            logger.info("Elasticsearch connection established.")
            return client
        except Exception as e:
            logger.info(
                f"Elasticsearch at {ELASTICSEARCH_URL} unavailable "
                f"({type(e).__name__}: {e}) — using processed_posts.json fallback. "
                f"Start the stack with: docker compose up -d"
            )
            return None

    # ── Data retrieval ─────────────────────────────────────────────────────

    def retrieve_post_documents(self) -> dict[str, PostDocument]:
        """
        Load post documents from Elasticsearch if available, otherwise from
        disk.
        """
        if self.elasticsearch_client:
            self.doc_index = retrieve_postdocs_from_elasticsearch(self.elasticsearch_client)
            return self.doc_index
        try:
            self.doc_index = retrieve_postdocs_from_disk(output_path=self.output_path)
        except FileNotFoundError as error:
            raise ProccessedPostsNotFoundError("Make sure you have run preprocessing.py") from error
        return self.doc_index

    # ── Model training ─────────────────────────────────────────────────────

    def train_topic_model(self) -> tuple[list[int], np.ndarray]:
        """
        Build and fit the BERTopic model; return topic assignments
        and probabilities.
        """
        logger.info("Training BERTopic model.")

        if not self.doc_index:
            raise RuntimeError("No post documents available for training.")

        texts, embeddings, post_ids = [], [], []
        for post_id, doc in self.doc_index.items():
            # for training extract the text as it was embedded
            # (i.e. cleaned and enriched with emojis converted to text, captions, etc.),
            text = extract_embedding_text(doc)
            # and the embedding from the document
            embedding = doc.doc_embedding
            if not embedding:
                continue
            texts.append(text)
            embeddings.append(embedding)
            post_ids.append(post_id)

        self._post_ids_for_training = post_ids
        self._texts_for_training = texts
        # Embeddings were converted to lists for JSON serialization when stored in ES;
        # convert back to arrays for training.
        embeddings_np = np.array(embeddings, dtype=np.float32)
        logger.info(f"Training on {len(texts)} posts.")

        ### EXERCISE ###
        # Review the parameters passed to the BERTopic constructor below. These
        # are the default parameters. Try changing some of them and see how it
        # affects the resulting topics.

        # Objective: Tune the model to find the 7 distinct topics created by
        # generator_posts.py.

        umap_model = UMAP(
            n_neighbors=15,  # local vs global balance
            n_components=2,  # output dimensions
            metric="euclidean",  # distance in input space
            output_metric="euclidean",  # distance in reduced space
            min_dist=0.1,  # default minimum spacing in projection
            random_state=RANDOM_SEED,  # reproducibility
            # Other UMAP parameters can be tuned as well, but these are the most
            # impactful for topic modeling results.
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=5,  # smallest cluster size
            min_samples=None,  # None -> uses min_cluster_size
            metric="euclidean",  # distance in UMAP space; Hint: Match UMAP.output_metric above.
            cluster_selection_method="eom",  # stable cluster extraction; alternative: 'leaf'
            allow_single_cluster=False,  # avoid one giant cluster
            #  Not the default, required by BERTopic for downstream visualizations and analyses.
            prediction_data=True,  # needed for BERTopic probabilities; Do not change.
        )
        # Used for topic word extraction after clustering.
        vectorizer = CountVectorizer(  # nosec B106
            input="content",  # Pass raw preprocessed text directly to the vectorizer
            encoding="utf-8",  # text encoding
            decode_error="strict",  # fail on bad decode
            strip_accents=None,  # keep accents
            lowercase=True,  # lowercase tokens
            # We have already preprocessed the text, but if we wanted to add
            # additional steps we could call a function here
            preprocessor=None,  # optional custom preprocessor
            # If we wanted to customize tokenization we could call a function here
            tokenizer=None,  # optional custom tokenizer
            stop_words=None,  # no stop-word list
            token_pattern=r"(?u)\b\w\w+\b",  # token regex (2+ chars)  # noqa: B106
            ngram_range=(1, 1),  # unigrams only
            analyzer="word",  # analyze at word level
            max_df=1.0,  # keep very common terms
            min_df=1,  # keep rare terms
            max_features=None,  # no vocab cap
            # We can also provide a custom vocabulary of words to consider for
            # topic modeling. By default it uses all words that appear in the corpus.
            vocabulary=None,  # learn vocabulary from corpus
            binary=False,  # use term counts, not binary
            dtype=np.int64,  # matrix integer type
        )
        # Used for topic labeling (represenantation) and generating the
        # topic_keyword_embeddings
        # See: notebooks/O5_noise.ipynb for an explanation of KeyBERT
        keybert_model = KeyBERTInspired(
            top_n_words=10,  # final words kept per topic
            nr_repr_docs=5,  # representative docs used per topic
            nr_samples=500,  # candidate docs sampled per topic
            nr_candidate_words=100,  # candidate words considered per topic
            random_state=RANDOM_SEED,  # reproducibility
        )
        # LLM-based labels. Uses Ollama if available, otherwise OpenAI API if
        # OPENAI_API_KEY is set, otherwise falls back to KeyBERT keywords.
        llm_model: BertTopicOpenAI | None = build_llm_representation(prompt=LABEL_PROMPT)
        representation = {"KeyBERT": keybert_model}
        if llm_model:
            representation["LLM"] = llm_model

        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,  # document embedding model
            umap_model=umap_model,  # dimensionality reduction model
            hdbscan_model=hdbscan_model,  # clustering model
            vectorizer_model=vectorizer,  # topic vocabulary model
            ctfidf_model=ClassTfidfTransformer(),  # class-based TF-IDF reweighting
            representation_model=representation,  # topic labeling models
            # Sets HDBSCAN's min_cluster_size. If not set above.
            # min_topic_size=10,  # Ignore.
            # Sets CountVectorizer's ngram_range. If not set above.
            # n_gram_range=(1, 1), # Ignore.
            top_n_words=10,  # words returned per topic
            # Required for topic evaluation
            calculate_probabilities=True,  # return per-topic probabilities; Do not change.
            verbose=True,  # log BERTopic progress
        )

        self.topics, self.probabilities = self.topic_model.fit_transform(texts, embeddings_np)

        logger.info("BERTopic training complete.")
        logger.info("Found %d topics", len(self.topic_model.get_topic_info()))

        # BERTopic reserves topic ID -1 for outliers,
        # So the number of true topics is max ID + 1
        self.n_topics = self.topic_model.get_topic_info().shape[0] - 1
        if self.n_topics == 0:
            logger.warning(
                "No valid topics found — all documents assigned to outlier class. "
                "Try adjusting BERTopic parameters (e.g. min_cluster_size, min_samples)."
            )
        return self.topics, self.probabilities

    # ── Artifact storage ───────────────────────────────────────────────────

    def store_model_data(self) -> None:
        """Orchestrate saving of all model artifacts to output_path."""
        if not self.topic_model:
            raise RuntimeError("Call train_topic_model() first.")

        valid_ids = [int(t) for t in self.topic_model.get_topic_info()["Topic"] if int(t) != -1]
        keywords_by_topic = {
            topic_id: [word for word, _ in self.topic_model.get_topic(topic_id)[:10]]
            for topic_id in valid_ids
        }
        self._save_model()
        self._save_assignments()
        self._save_labels(valid_ids, keywords_by_topic)
        self._save_topic_embeddings(valid_ids)
        self._save_keyword_embeddings(keywords_by_topic)
        self._save_topic_information()
        logger.info(f"All artifacts saved to {self.output_path}/")

    def _save_model(self) -> None:
        """Save BERTopic model binary to disk."""

        if not self.topic_model:
            raise RuntimeError("No topic model to save. Call train_topic_model() first.")
        if self.n_topics == 0:
            logger.warning("No valid topics found. Storing model will fail. Skipping model save.")
            return
        model_path = str(self.output_path / "bertopic_model")
        os.makedirs(model_path, exist_ok=True)
        self.topic_model.save(
            path=model_path,
            serialization="pytorch",
            save_ctfidf=True,
            save_embedding_model=self.embedding_model,
        )

    def _save_assignments(self) -> None:
        """Save topic assignments CSV and raw topics/probabilities JSON."""
        pd.DataFrame(
            {
                "post_id": self._post_ids_for_training,
                "text": self._texts_for_training,
                "topic_id": self.topics,
            }
        ).to_csv(self.output_path / "topic_assignments.csv", index=False)

        with open(self.output_path / "topics.json", "w") as f:
            json.dump(self.topics, f)
        with open(self.output_path / "probabilities.json", "w") as f:
            json.dump(self.probabilities.tolist(), f)

    def _save_labels(self, valid_ids: list[int], keywords_by_topic: dict[int, list[str]]) -> None:
        """Parse LLM labels (falling back to top keywords) and write topic_labels.json."""
        llm_labels: dict[int, str] = {}
        aspects = getattr(self.topic_model, "topic_aspects_", {}) or {}
        if "LLM" in aspects:
            for topic_id, label_list in aspects["LLM"].items():
                if not label_list:
                    continue
                first = label_list[0]
                # BERTopic returns either a plain string or a (str, score) tuple
                raw = first[0] if isinstance(first, (list, tuple)) else str(first)
                raw = re.sub(r"^topic\s*:\s*", "", raw, flags=re.IGNORECASE).strip()
                llm_labels[int(topic_id)] = raw

        labels_out: dict[str, dict] = {}
        for topic_id in sorted(valid_ids):
            topic_keywords = keywords_by_topic[topic_id]
            label = llm_labels.get(topic_id) or " · ".join(topic_keywords[:3])
            labels_out[str(topic_id)] = {"label": label}
            logger.info(f"  Topic {topic_id:2d}: {label}  [{', '.join(topic_keywords[:5])}]")

        with open(self.output_path / "topic_labels.json", "w") as f:
            json.dump(labels_out, f, indent=2, ensure_ascii=False)

    def _save_topic_embeddings(self, valid_ids: list[int]) -> None:
        """Write topic_embeddings.json from BERTopic's internal topic embedding matrix."""
        if not hasattr(self.topic_model, "topic_embeddings_"):
            raise AttributeError(
                "topic_embeddings_ not found on topic model. "
                "Check that the model is fitted before calling store_model_data()."
            )
        topic_embeddings = self.topic_model.topic_embeddings_
        topic_info = self.topic_model.get_topic_info()

        # BERTopic's topic_embeddings_ array is positional, not keyed by topic id.
        # Build a id->embedding mapping using topic_info row order.
        topic_ids_in_order = [int(t) for t in topic_info["Topic"].tolist()]

        if len(topic_embeddings) != len(topic_ids_in_order):
            raise ValueError(
                f"topic_embeddings_ length ({len(topic_embeddings)}) does not match "
                f"topic_info Topic count ({len(topic_ids_in_order)}). "
                "BERTopic internal state may be corrupted."
            )

        topic_embedding_map: dict[int, list[float]] = {
            topic_id: embedding.tolist()
            for topic_id, embedding in zip(topic_ids_in_order, topic_embeddings, strict=True)
            if topic_id != -1
        }

        # Keep only valid non-outlier topics
        topic_embedding_map = {
            topic_id: topic_embedding_map[topic_id]
            for topic_id in valid_ids
            if topic_id in topic_embedding_map
        }
        with open(self.output_path / "topic_embeddings.json", "w") as f:
            json.dump(
                {str(topic_id): embedding for topic_id, embedding in topic_embedding_map.items()},
                f,
            )

    def _save_keyword_embeddings(
        self, keywords_by_topic: dict[int, list[str]]
    ) -> dict[int, list[str]]:
        """Write topic_keyword_embeddings.json using KeyBERT representation terms."""
        _ = keywords_by_topic  # Kept for signature compatibility with current call sites.

        if not self.topic_model.topic_aspects_:
            raise AttributeError(
                "topic_aspects_ not found on topic model. "
                "Check that the model is fitted before calling store_model_data()."
            )

        aspects = self.topic_model.topic_aspects_
        if not isinstance(aspects, dict) or not aspects.get("KeyBERT"):
            raise AttributeError(
                "KeyBERT topic aspects not found on topic model. "
                "Check representation_model and confirm KeyBERT is enabled."
            )
        keybert_aspects = aspects["KeyBERT"]

        top_keywords_by_topic: dict[int, list[str]] = {}
        for topic_id, raw_terms in keybert_aspects.items():
            tokens: list[str] = []
            for item in raw_terms:
                token = item[0] if isinstance(item, (list, tuple)) and item else str(item)
                token = str(token).strip()
                if token:
                    tokens.append(token)
            if tokens:
                top_keywords_by_topic[int(topic_id)] = tokens[:TOP_N_SEARCH_EMBEDDING_KEYWORDS]

        topic_keyword_embs = {
            topic_id: self.embedding_model.encode(
                " ".join(topic_keywords), convert_to_numpy=True
            ).tolist()
            for topic_id, topic_keywords in top_keywords_by_topic.items()
        }
        with open(self.output_path / "topic_keyword_embeddings.json", "w") as f:
            json.dump(
                {str(topic_id): embedding for topic_id, embedding in topic_keyword_embs.items()}, f
            )
        return top_keywords_by_topic

    def _save_topic_information(self) -> pd.DataFrame:
        """Write topic_information.json with metadata about each topic."""
        if self.topic_model.get_topic_info() is None:
            raise AttributeError(
                "topic_info_ not found on topic model. "
                "Check that the model is fitted before calling store_model_data()."
            )
        # Access topic_info_ directly to ensure topic -1 (outliers) is included;
        topic_info = self.topic_model.get_topic_info()
        topic_info.to_csv(path_or_buf=self.output_path / "topic_information.csv", index=False)
        return topic_info

    def generate_visualizations(self) -> None:
        """Write BERTopic HTML visualizations to output_path."""
        if not self.topic_model or self.n_topics == 0:
            logger.warning(
                "No topic model or valid topics available for visualization."
                " Skipping visualization."
            )
            return
        out = str(self.output_path)
        self.topic_model.visualize_topics().write_html(f"{out}/topic_visualization.html")
        self.topic_model.visualize_barchart().write_html(f"{out}/barchart.html")
        self.topic_model.visualize_heatmap().write_html(f"{out}/heatmap.html")
        self.topic_model.visualize_hierarchy().write_html(f"{out}/hierarchy.html")
        self.topic_model.visualize_term_rank().write_html(f"{out}/term_rank.html")
        logger.info("Visualizations saved.")

    def load_topic_model(self) -> BERTopic:
        """Load a previously saved BERTopic model from output_path."""
        model_path = str(self.output_path / "bertopic_model")
        self.topic_model = BERTopic.load(model_path)
        logger.info(f"Model loaded from {model_path}.")
        return self.topic_model

    # ── Main pipeline ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Run the full topic modeling pipeline: load → train → save → visualize."""
        self.retrieve_post_documents()
        self.train_topic_model()
        self.store_model_data()
        self.generate_visualizations()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    TopicModeler().run()
