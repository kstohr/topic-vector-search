# Thinking of Topic Modeling as Search
Use vector search to search for documents related to topics derived from a topic model.

**PyCon 2026 Tutorial** · 3 hours · [Slide deck](https://docs.google.com/presentation/d/1hayVZQV9psYBdM6HwAjF3uWcQu7lieLP/edit)

---

## Workshop Quick Start

**Do this before the workshop** — it requires a ~2 GB download and can be slow on conference Wi-Fi.

### 1. Install dependencies

```bash
uv sync
```

### 2. Start Docker services (Elasticsearch + Ollama)

```bash
docker compose up -d
```

This starts two containers:
- **Elasticsearch** on port 9200 — vector database (optional, in-memory fallback provided)
- **Ollama** on port 11434 — local LLM for topic labeling, automatically pulls `qwen2.5:3b` (~1.9 GB)

> **No Docker?** All exercises still work. Search uses an in-memory fallback and topic labels
> fall back to keyword phrases. See the [No Docker path](#no-docker-no-problem) below.

### 3. Verify your setup

```bash
uv run jupyter lab notebooks/00_setup_check.ipynb
```

### 4. Start the demo app

```bash
uv run streamlit run app.py
```

---

## No Docker? No problem.

All exercises work without Docker:

- **Search**: `InMemorySearcher` in `src/search.py` is used automatically when
  Elasticsearch is unavailable.

- **Topic labels**: the pipeline falls back to KeyBERT keyword phrases if Ollama is not running.
  If you have an OpenAI key, set `OPENAI_API_KEY` in `.env` and it will be used instead.

---

## Topic Label LLM Priority

The topic modeling pipeline (`src/topic_model.py`) picks a labeling backend in this order:

| Priority | Condition | Model |
|---|---|---|
| 1 | Ollama running at `localhost:11434` | `qwen2.5:3b` (local, no key needed) |
| 2 | `OPENAI_API_KEY` set in environment | `gpt-4o-mini` via OpenAI API |
| 3 | Neither | KeyBERT keyword phrases |

To override the Ollama model or URL:

```bash
OLLAMA_MODEL=llama3.2:3b uv run python exercises/02_topic_model.py
```

---

## Exercises

| # | File | Topic |
|---|------|-------|
| 1 | `notebooks/01_embeddings.ipynb` | Creating document embeddings |
| 2 | `exercises/02_topic_model.py` | Building the topic model pipeline |
| 3 | `notebooks/03_search_evaluation.ipynb` | Evaluating topic vs. localized embeddings |

Solutions are in `solutions/`.

---

## Environment Variables (optional)

Create a `.env` file to configure OpenAI as a labeling backend:

```
OPENAI_API_KEY="sk-..."
OPENAI_ORGANIZATION="org-..."
OPENAI_PROJECT="proj_..."
```

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker + Docker Compose *(optional)*

---

# Presentation

Typically when you think about using a topic model in production you encounter two hurdles:
First, topics change continually, and document tags become stale as soon as they are created.
Second, while unsupervised topic models do a good job of clustering topics, creating robust,
human-interpretable labels is challenging. Framing topic modeling as a search problem helps
overcome these challenges and makes it easier to use supervised or unsupervised topic models
in real-time applications.

- [PyBay 2024 - Thinking of Topic Modeling as Search (video)](https://www.youtube.com/watch?v=vymhlfxAd4Y)
- [Slide deck](https://docs.google.com/presentation/d/1UfaDLzG9WvTeP8I64za-ycwnst8bXMzx/edit?usp=sharing)

---

# References and Credits

Thanks to Maarten Grootendorst for [BERTopic](https://github.com/MaartenGr/BERTopic) and
to [Ray 'Urgent' McLendon](https://www.linkedin.com/in/raymclendon/) for his interest and input.

- [Elasticsearch Vector Search](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html)
- [An Intuitive Introduction to Text Embeddings](https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/)
- [BERTopic](https://github.com/MaartenGr/BERTopic)
- [Comparing Clustering Algorithms (HDBSCAN)](https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html)
- [c-TF-IDF](https://www.maartengrootendorst.com/blog/ctfidf/)
- [Text Search vs Vector Search](https://towardsdatascience.com/text-search-vs-vector-search-better-together-3bd48eb6132a)
