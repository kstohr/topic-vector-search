"""
Topic Vector Search — Workshop Demo
Run with: streamlit run app.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent / "src"))
from search import InMemorySearcher  # noqa: E402

OUTPUT = Path(__file__).parent / "output"
REPO = Path(__file__).parent

st.set_page_config(page_title="Topic Vector Search", layout="wide")

# ---------------------------------------------------------------------------
# Custom styles
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.topic-card {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    cursor: pointer;
    transition: box-shadow 0.2s;
}
.topic-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
.topic-emoji { font-size: 2rem; }
.topic-label { font-weight: 600; margin: 6px 0 4px; }
.topic-stat { color: #666; font-size: 0.85rem; }
.trending-bar { font-size: 0.75rem; color: #999; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (all cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_topic_keywords() -> dict[int, list[str]]:
    with open(OUTPUT / "bertopic_model" / "topics.json") as f:
        data = json.load(f)
    reps = data.get("topic_representations", {})
    return {int(k): [w for w, _ in v] for k, v in reps.items() if int(k) != -1}


@st.cache_data
def load_topic_labels() -> dict[int, dict]:
    with open(OUTPUT / "topic_labels.json") as f:
        return {int(k): v for k, v in json.load(f).items()}


@st.cache_data
def load_assignments() -> pd.DataFrame:
    return pd.read_csv(OUTPUT / "topic_assignments.csv")


@st.cache_data
def load_posts_with_likes() -> dict[str, dict]:
    """Load sample_posts for likes + created_at metadata, keyed by post_id."""
    with open(REPO / "sample_posts.json") as f:
        posts = json.load(f)
    return {p["post_id"]: p for p in posts}


@st.cache_data
def load_doc_index() -> list[dict]:
    """Load doc_index (posts with embeddings) as a list."""
    with open(OUTPUT / "doc_index.json") as f:
        return list(json.load(f).values())


@st.cache_resource
def build_searcher() -> InMemorySearcher:
    return InMemorySearcher(load_doc_index())


@st.cache_resource
def build_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def build_localized_embeddings() -> dict[int, list[float]]:
    """Compute keyword-based topic embeddings. Saves to disk on first run."""
    path = OUTPUT / "topic_embeddings_localized.json"
    if path.exists():
        with open(path) as f:
            return {int(k): v for k, v in json.load(f).items()}
    model = build_embedding_model()
    keywords = load_topic_keywords()
    result = {}
    for tid, words in keywords.items():
        emb = model.encode(" ".join(words[:10]), convert_to_numpy=True)
        result[tid] = emb.tolist()
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in result.items()}, f)
    return result


@st.cache_data
def compute_trending() -> list[dict]:
    """Rank topics by total likes + recency. Returns top 3."""
    assignments = load_assignments()
    likes_data = load_posts_with_likes()
    labels = load_topic_labels()
    keywords = load_topic_keywords()

    def parse_ts(raw: str) -> float:
        """Parse ISO datetime string → UTC unix timestamp (always naive-safe)."""
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()

    all_ts = [parse_ts(p["created_at"]) for p in likes_data.values()]
    most_recent_ts = max(all_ts)

    rows = []
    for topic_id in sorted(labels.keys()):
        post_ids = assignments.loc[assignments["topic_id"] == topic_id, "post_id"].tolist()
        if not post_ids:
            continue

        total_likes = sum(likes_data.get(pid, {}).get("likes", 0) for pid in post_ids)

        # Recency: average days ago relative to most recent post
        ts_list = []
        for pid in post_ids:
            raw = likes_data.get(pid, {}).get("created_at", "")
            if raw:
                try:
                    ts_list.append(parse_ts(raw))
                except ValueError:
                    pass

        if ts_list:
            avg_ts = sum(ts_list) / len(ts_list)
            avg_days_ago = (most_recent_ts - avg_ts) / 86400
        else:
            avg_days_ago = 999
        # Normalize recency to 0–100 (most recent = 100)
        max_days = 365
        recency_score = max(0, 100 - (avg_days_ago / max_days) * 100)

        # Combined trending score
        trending_score = total_likes * 0.65 + recency_score * 0.35

        info = labels[topic_id]
        rows.append({
            "topic_id": topic_id,
            "label": info["label"],
            "emoji": info["emoji"],
            "total_likes": total_likes,
            "post_count": len(post_ids),
            "recency_score": round(recency_score, 1),
            "trending_score": trending_score,
            "keywords": keywords.get(topic_id, [])[:5],
        })

    rows.sort(key=lambda r: r["trending_score"], reverse=True)
    return rows[:3]


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def search_by_keywords(query: str, top_k: int = 20) -> pd.DataFrame:
    model = build_embedding_model()
    searcher = build_searcher()
    embedding = model.encode(query, convert_to_numpy=True)
    results = searcher.search_similar_documents(embedding, top_k=top_k)
    return _results_to_df(results)


def search_by_topic(topic_id: int, top_k: int = 20) -> pd.DataFrame:
    searcher = build_searcher()
    localized = build_localized_embeddings()
    embedding = np.array(localized[topic_id], dtype=np.float32)
    results = searcher.search_similar_documents(embedding, top_k=top_k)
    return _results_to_df(results)


def _results_to_df(results: list[dict]) -> pd.DataFrame:
    assignments = load_assignments()
    labels = load_topic_labels()
    id_to_label = {tid: f"{v['emoji']} {v['label']}" for tid, v in labels.items()}

    rows = []
    for r in results:
        assigned_id = assignments.loc[
            assignments["post_id"] == r["post_id"], "topic_id"
        ]
        topic_id = int(assigned_id.values[0]) if len(assigned_id) else -1
        rows.append({
            "score": round(r["score"], 3),
            "post": r["post_text"],
            "topic": id_to_label.get(topic_id, f"Topic {topic_id}"),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "active_topic" not in st.session_state:
    st.session_state.active_topic = None   # topic_id int or None
if "keyword_query" not in st.session_state:
    st.session_state.keyword_query = ""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("Topic Vector Search")
st.caption("Search the document store using keywords or click a trending topic.")

# — Search bar —
query_input = st.text_input(
    label="search",
    placeholder="Search posts by keywords…",
    value=st.session_state.keyword_query,
    label_visibility="collapsed",
)
if query_input != st.session_state.keyword_query:
    st.session_state.keyword_query = query_input
    st.session_state.active_topic = None  # keyword search clears topic selection

st.markdown("---")

# — Trending topics —
st.subheader("Trending")
trending = compute_trending()
cols = st.columns(3)

for col, topic in zip(cols, trending):
    with col:
        label_with_emoji = f"{topic['emoji']} {topic['label']}"
        kw_str = " · ".join(topic["keywords"])
        st.markdown(f"""
<div class="topic-card">
  <div class="topic-emoji">{topic['emoji']}</div>
  <div class="topic-label">{topic['label']}</div>
  <div class="topic-stat">❤️ {topic['total_likes']:,} likes &nbsp;·&nbsp; {topic['post_count']} posts</div>
  <div class="trending-bar">{kw_str}</div>
</div>
""", unsafe_allow_html=True)
        if st.button("Search this topic", key=f"btn_{topic['topic_id']}"):
            st.session_state.active_topic = topic["topic_id"]
            st.session_state.keyword_query = ""
            st.rerun()

st.markdown("---")

# — Results —
if st.session_state.active_topic is not None:
    topic_id = st.session_state.active_topic
    info = load_topic_labels().get(topic_id, {})
    label = f"{info.get('emoji', '')} {info.get('label', f'Topic {topic_id}')}"
    st.subheader(f"Results for {label}")
    with st.spinner("Searching…"):
        df = search_by_topic(topic_id)
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "score": st.column_config.NumberColumn("Score", format="%.3f", width="small"),
            "post": st.column_config.TextColumn("Post", width="large"),
            "topic": st.column_config.TextColumn("Topic", width="medium"),
        },
        hide_index=True,
    )

elif st.session_state.keyword_query.strip():
    query = st.session_state.keyword_query.strip()
    st.subheader(f'Results for "{query}"')
    with st.spinner("Searching…"):
        df = search_by_keywords(query)
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "score": st.column_config.NumberColumn("Score", format="%.3f", width="small"),
            "post": st.column_config.TextColumn("Post", width="large"),
            "topic": st.column_config.TextColumn("Topic", width="medium"),
        },
        hide_index=True,
    )

else:
    st.markdown(
        "<p style='color:#999; text-align:center; padding: 2rem 0;'>"
        "Enter keywords above or click a trending topic to search."
        "</p>",
        unsafe_allow_html=True,
    )
