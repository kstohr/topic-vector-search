"""
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
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from src.config import (
    ELASTICSEARCH_URL,
    OLLAMA_MODEL,
    OLLAMA_URL,
    OPENAI_MODEL,
)
from src.config import (
    EMBEDDING_MODEL_NAME as EMBEDDING_MODEL,
)
from src.models import PostDocument
from src.preprocess import embedding_text

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


def _build_llm_representation() -> BertTopicOpenAI | None:
    """
    Return a BERTopic OpenAI representation model pointed at the best available backend:
      1. Ollama (local, no key needed)
      2. OpenAI (OPENAI_API_KEY env var)
      3. None → KeyBERT-only labels
    """
    import httpx
    from openai import OpenAI

    try:
        r = httpx.get(OLLAMA_URL.replace("/v1", "/api/tags"), timeout=2)
        if r.status_code == 200:
            logger.info(f"Ollama reachable at {OLLAMA_URL} — using {OLLAMA_MODEL}")
            client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
            return BertTopicOpenAI(
                client=client,
                model=OLLAMA_MODEL,
                exponential_backoff=True,
                chat=True,
                prompt=LABEL_PROMPT,
                nr_docs=5,
            )
    except Exception:
        pass

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        logger.info("Using OpenAI API for topic labels.")
        client = OpenAI(
            api_key=api_key,
            organization=os.getenv("OPENAI_ORGANIZATION"),
            project=os.getenv("OPENAI_PROJECT"),
        )
        return BertTopicOpenAI(
            client=client,
            model=OPENAI_MODEL,
            exponential_backoff=True,
            chat=True,
            prompt=LABEL_PROMPT,
            nr_docs=5,
        )

    logger.warning("No LLM available — topic labels will use KeyBERT keywords only.")
    return None


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

        self.elasticsearch_client = self._try_elasticsearch()

    def _try_elasticsearch(self):
        """Return a connected Elasticsearch client, or None if unavailable."""
        try:
            from elasticsearch import Elasticsearch

            client = Elasticsearch(ELASTICSEARCH_URL)
            client.info()
            logger.info("Elasticsearch connection established.")
            return client
        except Exception:
            logger.info("Elasticsearch not available — will use processed_posts.json fallback.")
            return None

    # ── Data retrieval ─────────────────────────────────────────────────────

    def retrieve_post_documents(self) -> dict[str, PostDocument]:
        """Load post documents from Elasticsearch if available, otherwise from disk."""
        if self.elasticsearch_client:
            return self._retrieve_from_elasticsearch()
        return self._retrieve_from_disk()

    def _retrieve_from_elasticsearch(self) -> dict[str, PostDocument]:
        """Scroll through Elasticsearch to retrieve all post documents."""
        if not self.elasticsearch_client.indices.exists(index=self.index_name):
            logger.warning(
                f"Elasticsearch index '{self.index_name}' does not exist. "
                "Run 'uv run python -m src.preprocess' first, or falling back to disk."
            )
            return self._retrieve_from_disk()

        logger.info("Retrieving post documents from Elasticsearch.")
        response = self.elasticsearch_client.search(
            index=self.index_name,
            scroll="2m",
            body={"size": 500, "query": {"match_all": {}}},
        )
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]
        while hits:
            for hit in hits:
                post = PostDocument(**hit["_source"])
                self.doc_index[post.post_id] = post
            response = self.elasticsearch_client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = response["_scroll_id"]
            hits = response["hits"]["hits"]
        self.elasticsearch_client.clear_scroll(scroll_id=scroll_id)
        logger.info(f"Retrieved {len(self.doc_index)} posts from Elasticsearch.")
        return self.doc_index

    def _retrieve_from_disk(self) -> dict[str, PostDocument]:
        """Load post documents from output/processed_posts.json."""
        path = self.output_path / "processed_posts.json"
        logger.info(f"Loading posts from {path}.")
        with open(path) as f:
            raw = json.load(f)
        for post_id, data in raw.items():
            self.doc_index[post_id] = PostDocument(**data)
        logger.info(f"Loaded {len(self.doc_index)} posts from disk.")
        return self.doc_index

    # ── Model training ─────────────────────────────────────────────────────

    def train_topic_model(self) -> tuple[list[int], np.ndarray]:
        """Build and fit the BERTopic model; return topic assignments and probabilities."""
        logger.info("Training BERTopic model.")

        texts, embeddings, post_ids = [], [], []
        for post_id, doc in self.doc_index.items():
            raw = doc.model_dump(mode="json")
            text = embedding_text(raw)
            emb = raw.get("doc_embedding", [])
            if not text or not emb:
                continue
            texts.append(text)
            embeddings.append(emb)
            post_ids.append(post_id)

        self._post_ids_for_training = post_ids
        self._texts_for_training = texts
        embeddings_np = np.array(embeddings, dtype=np.float32)
        logger.info(f"Training on {len(texts)} posts.")

        vectorizer = CountVectorizer(
            min_df=2,
            ngram_range=(1, 3),
            stop_words="english",
        )
        keybert_model = KeyBERTInspired(
            top_n_words=10,
            nr_repr_docs=5,
            nr_samples=500,
            nr_candidate_words=100,
            random_state=RANDOM_SEED,
        )
        llm_model = _build_llm_representation()
        representation = {"KeyBERT": keybert_model}
        if llm_model:
            representation["LLM"] = llm_model

        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            vectorizer_model=vectorizer,
            ctfidf_model=ClassTfidfTransformer(),
            representation_model=representation,
            min_topic_size=10,
            n_gram_range=(1, 3),
            top_n_words=10,
            calculate_probabilities=True,
            verbose=True,
        )

        self.topics, self.probabilities = self.topic_model.fit_transform(texts, embeddings_np)

        n = self.topic_model.get_topic_info().shape[0] - 1
        logger.info(f"Found {n} topics.")
        return self.topics, self.probabilities

    # ── Artifact storage ───────────────────────────────────────────────────

    def store_model_data(self) -> None:
        """Orchestrate saving of all model artifacts to output_path."""
        if not self.topic_model:
            raise RuntimeError("Call train_topic_model() first.")

        valid_ids = [int(t) for t in self.topic_model.get_topic_info()["Topic"] if int(t) != -1]
        keywords_by_topic = {
            tid: [w for w, _ in self.topic_model.get_topic(tid)[:10]] for tid in valid_ids
        }

        self._save_model()
        self._save_assignments()
        self._save_labels(valid_ids, keywords_by_topic)
        self._save_topic_embeddings(valid_ids)
        self._save_keyword_embeddings(keywords_by_topic)
        logger.info(f"All artifacts saved to {self.output_path}/")

    def _save_model(self) -> None:
        """Save BERTopic model binary to disk."""
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
            for tid, label_list in aspects["LLM"].items():
                if not label_list:
                    continue
                first = label_list[0]
                # BERTopic returns either a plain string or a (str, score) tuple
                raw = first[0] if isinstance(first, (list, tuple)) else str(first)
                raw = re.sub(r"^topic\s*:\s*", "", raw, flags=re.IGNORECASE).strip()
                llm_labels[int(tid)] = raw

        labels_out: dict[str, dict] = {}
        for tid in sorted(valid_ids):
            kws = keywords_by_topic[tid]
            label = llm_labels.get(tid) or " · ".join(kws[:3])
            labels_out[str(tid)] = {"label": label}
            logger.info(f"  Topic {tid:2d}: {label}  [{', '.join(kws[:5])}]")

        with open(self.output_path / "topic_labels.json", "w") as f:
            json.dump(labels_out, f, indent=2, ensure_ascii=False)

    def _save_topic_embeddings(self, valid_ids: list[int]) -> None:
        """Write topic_embeddings.json from BERTopic's internal topic embedding matrix."""
        # index 0 = outlier cluster, real topics start at 1
        topic_embs = getattr(self.topic_model, "topic_embeddings_", None)
        if topic_embs is None:
            return
        topic_embedding_map = {
            tid: topic_embs[tid + 1].tolist()
            for tid in valid_ids
            if (tid + 1) < len(topic_embs)
        }
        with open(self.output_path / "topic_embeddings.json", "w") as f:
            json.dump({str(k): v for k, v in topic_embedding_map.items()}, f)

    def _save_keyword_embeddings(self, keywords_by_topic: dict[int, list[str]]) -> None:
        """Write topic_keyword_embeddings.json by encoding each topic's top keywords."""
        topic_keyword_embs = {
            tid: self.embedding_model.encode(" ".join(kws[:10]), convert_to_numpy=True).tolist()
            for tid, kws in keywords_by_topic.items()
        }
        with open(self.output_path / "topic_keyword_embeddings.json", "w") as f:
            json.dump({str(k): v for k, v in topic_keyword_embs.items()}, f)

    def generate_visualizations(self) -> None:
        """Write BERTopic HTML visualizations to output_path."""
        if not self.topic_model:
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
        """Run the full topic modeling pipeline: load → train → visualize → save."""
        self.retrieve_post_documents()
        self.train_topic_model()
        self.generate_visualizations()
        self.store_model_data()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    TopicModeler().run()
