# Thinking of Topic Modeling as Search
Use vector search to search for documents related to topics derived from a topic model.

**PyCon 2026 Tutorial** · 3 hours · [Slide
deck](https://docs.google.com/presentation/d/1hayVZQV9psYBdM6HwAjF3uWcQu7lieLP/edit)


# Presentation

Typically when you think about using a topic model in production you encounter two hurdles:
First, topics change continually, and document tags become stale as soon as they are created.
Second, while unsupervised topic models do a good job of clustering topics, creating robust,
human-interpretable labels is challenging. Framing topic modeling as a search problem helps
overcome these challenges and makes it easier to use supervised or unsupervised topic models
in real-time applications.

- [PyBay 2024 - Thinking of Topic Modeling as Search
  (video)](https://www.youtube.com/watch?v=vymhlfxAd4Y)

---
## Requirements

- [Python 3.12+](https://www.python.org/downloads/)
- [git](https://git-scm.com/downloads) — git cli to clone the repo
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — package manager
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — includes Docker Engine + Docker Compose for Mac, Windows, and Linux
- **Machine resources**: ~20 GB free disk and 8 GB RAM (workshop downloads ~13 GB total). On Mac/Windows, ensure Docker Desktop has at least 4 GB allocated under Settings → Resources.


---

## Workshop Quick Start

**Do this before the workshop** — items #1 and #2 requires at least ~15 GB
download and can be very slow on conference Wi-Fi with many users.

Alternately: Use Github Codespace. Instructions are here: 

[TROUBLESHOOTING.md "Setup Alternative: Github Codespace"](TROUBLESHOOTING.md#setup-alternative-github-codespace)

### 1. Install dependencies (~6 GB)

`uv` is the package manager this repo uses.

Mac and Linux:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:
```
uv --version
```

```bash
uv sync
```

### 2. Start Docker services (Elasticsearch + Ollama) (~7 GB)

```bash
docker compose up -d
```

This starts two containers:
- **Elasticsearch** on port 9201 — vector database (optional, in-memory fallback provided) (~1 GB)
- **Ollama** on port 11434 — local LLM for topic labeling, automatically pulls `qwen2.5:3b` (~8GB: ~6 GB for docker + ~1.9 GB for qwen)

> **No Docker?** All code should still work. Search uses an in-memory fallback and topic labels
> fall back to keyword phrases. See the [No Docker path](#no-docker-no-problem) below.

### 3. Verify your setup

```bash
uv run jupyter lab notebooks/00_setup_check.ipynb
```

### 4. Start the demo app

```bash
uv run streamlit run app.py
```

### 5. Shutdown the demo app and docker containers 
```bash

# Control-C in the running terminal, or in a different terminal: 
pkill -f "streamlit run app.py" 
docker compose down 
-- make any changes necessary -- 
uv sync 
uv run streamlit run app.py # the app preloads sample posts on startup
```
---

## No Docker? No problem.

Source code can run without Docker:

- **Search**: `InMemorySemanticSearcher` and `InMemoryKeywordSearcher` in `src/search.py`
  are used automatically when Elasticsearch is unavailable.

- **Topic labels**: the pipeline falls back to KeyBERT keyword phrases if Ollama
  is not running. If you have an OpenAI key, set `OPENAI_API_KEY` in `.env` or your env vars, and
  it will be used instead.

---

## Topic Label LLM 

The topic modeling pipeline (`src/topic_model.py`) uses an LLM to label the
topics.  It picks a labeling backend in this order:

| Priority | Condition | Model |
|---|---|---|
| 1 | Ollama running at `localhost:11434` | `qwen2.5:3b` (local, no key needed) |
| 2 | `OPENAI_API_KEY` set in environment | `gpt-4o-mini` via OpenAI API |
| 3 | Neither | KeyBERT keyword phrases |

To override the Ollama model or URL:

```bash
OLLAMA_MODEL=llama3.2:3b uv run python src/topic_model.py
```

---

## Explanatory notebooks and Code Exercises


As the demo progresses you will update specific functions ("Exercises") in the
`/src` code based on concepts explained in the notebooks created for the workshop.
The `/src` code powers a simple Streamlit demo app. As you progress through the
workshop you will add functionality and evaluation tooling to the demo app. The 
intent is that this will be a no code/low code workshop where we focus on
learning the key concepts behind embeddings (dense vectors),topic modeling and search.  

The final demo `/src` code is stored in `solutions/` for reference or if you run
out of time on any given exercise. 

---

## Environment Variables (optional)

Create a `.env` file to configure OpenAI as a labeling backend:

```
OPENAI_API_KEY="sk-..."
```

`OPENAI_ORGANIZATION` and `OPENAI_PROJECT` are optional — only needed for enterprise or team accounts:

```
OPENAI_ORGANIZATION="org-..."   # enterprise/team only
OPENAI_PROJECT="proj_..."       # enterprise/team only
```

---

## Known Issues

**`Exception ignored in: ResourceTracker.__del__`** — you may see this message
in the terminal after running `uv run python -m src.topic_model`. It is a
[known bug](https://github.com/uqfoundation/multiprocess/issues) in
`multiprocess` It can be ignored.

---

# References and Credits

Thanks to Maarten Grootendorst for [BERTopic](https://github.com/MaartenGr/BERTopic), 
to [Ray 'Urgent' McLendon](https://www.linkedin.com/in/raymclendon/) for his
interest and input and [Chris Brousseau](https://www.linkedin.com/in/chrisbrousseau/) for making sure this
repo actually runs and the content is ... on-topic. 

- [Text Search vs Vector Search](https://towardsdatascience.com/text-search-vs-vector-search-better-together-3bd48eb6132a)
- [Elasticsearch Vector Search](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html)
- [An Intuitive Introduction to Text Embeddings](https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/)
- [BERTopic](https://github.com/MaartenGr/BERTopic)
- [Comparing Clustering Algorithms (HDBSCAN)](https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html)
- [c-TF-IDF](https://www.maartengrootendorst.com/blog/ctfidf/)
