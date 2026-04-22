"""
Topic Vector Search — Workshop Demo
Run with: uv run streamlit run app.py

The app adapts automatically as workshop exercises are completed:
  LO1  Lexical (LIKE) search · no trending topics
  LO2  Embedding search · no trending topics
  LO3  Embedding search · trending topics (topic-centroid embeddings)
  LO4  Embedding search · trending topics (localized embeddings) ← full solution
  LO5  Image posts appear in results via vision-model captions
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
from search import InMemorySearcher, LexicalSearcher  # noqa: E402

OUTPUT = Path(__file__).parent / "output"
REPO = Path(__file__).parent

st.set_page_config(page_title="Topic Vector Search", layout="wide")

st.markdown("""
<style>
.topic-card {
    border: 1px solid #e0e0e0; border-radius: 10px;
    padding: 16px; text-align: center;
}
.topic-emoji  { font-size: 2rem; }
.topic-label  { font-weight: 600; margin: 6px 0 4px; }
.topic-stat   { color: #666; font-size: 0.85rem; }
.topic-kw     { color: #999; font-size: 0.75rem; margin-top: 4px; }
.mode-badge   { font-size: 0.75rem; color: #888; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Capability detection ────────────────────────────────────────────────────

def _has_embeddings() -> bool:
    """True if doc_index.json exists and at least one post has a non-empty embedding."""
    path = OUTPUT / "doc_index.json"
    if not path.exists():
        return False
    try:
        with open(path) as f:
            index = json.load(f)
        sample = next(iter(index.values()), {})
        return len(sample.get("doc_embedding", [])) > 0
    except Exception:
        return False


def _has_topics() -> bool:
    path = OUTPUT / "topic_assignments.csv"
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
        return len(df) > 0 and df["topic_id"].nunique() > 1
    except Exception:
        return False


def _has_localized() -> bool:
    path = OUTPUT / "topic_embeddings_localized.json"
    if not path.exists():
        return False
    try:
        with open(path) as f:
            d = json.load(f)
        return len(d) > 0
    except Exception:
        return False


def _has_image_captions() -> bool:
    with open(REPO / "sample_posts.json") as f:
        posts = json.load(f)
    return any(p.get("image_caption") for p in posts)


# ── Data loading (all cached) ───────────────────────────────────────────────

@st.cache_data
def load_raw_posts() -> list[dict]:
    with open(REPO / "sample_posts.json") as f:
        return json.load(f)


@st.cache_data
def load_posts_by_id() -> dict[str, dict]:
    return {p["post_id"]: p for p in load_raw_posts()}


@st.cache_data
def load_doc_index() -> list[dict]:
    with open(OUTPUT / "doc_index.json") as f:
        return list(json.load(f).values())


@st.cache_data
def load_assignments() -> pd.DataFrame:
    return pd.read_csv(OUTPUT / "topic_assignments.csv")


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


@st.cache_resource
def build_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def build_searcher():
    if _has_embeddings():
        return InMemorySearcher(load_doc_index())
    return LexicalSearcher(load_raw_posts())


@st.cache_data
def build_localized_embeddings() -> dict[int, list[float]]:
    """Load from disk, or compute from keywords and save."""
    path = OUTPUT / "topic_embeddings_localized.json"
    if path.exists():
        with open(path) as f:
            return {int(k): v for k, v in json.load(f).items()}
    # Generate and save
    model = build_embedding_model()
    keywords = load_topic_keywords()
    result = {tid: model.encode(" ".join(ws[:10]), convert_to_numpy=True).tolist()
              for tid, ws in keywords.items()}
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in result.items()}, f)
    return result


@st.cache_data
def build_naive_embeddings() -> dict[int, list[float]]:
    """BERTopic centroid embeddings (used at LO3 before localized are available)."""
    from bertopic import BERTopic
    model = BERTopic.load(str(OUTPUT / "bertopic_model"))
    labels = load_topic_labels()
    return {tid: model.topic_embeddings_[tid + 1].tolist() for tid in labels}


@st.cache_data
def compute_trending() -> list[dict]:
    """Rank topics by aggregate likes + recency. Returns top 3."""
    assignments = load_assignments()
    posts_by_id = load_posts_by_id()
    labels = load_topic_labels()
    keywords = load_topic_keywords()

    def parse_ts(raw: str) -> float:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()

    most_recent_ts = max(parse_ts(p["created_at"]) for p in posts_by_id.values())

    rows = []
    for topic_id in sorted(labels.keys()):
        post_ids = assignments.loc[assignments["topic_id"] == topic_id, "post_id"].tolist()
        if not post_ids:
            continue
        total_likes = sum(posts_by_id.get(pid, {}).get("likes", 0) for pid in post_ids)
        ts_list = [
            parse_ts(posts_by_id[pid]["created_at"])
            for pid in post_ids if pid in posts_by_id
        ]
        avg_days_ago = (most_recent_ts - sum(ts_list) / len(ts_list)) / 86400 if ts_list else 999
        recency = max(0.0, 100.0 - (avg_days_ago / 365) * 100)
        trending_score = total_likes * 0.65 + recency * 0.35

        info = labels[topic_id]
        rows.append({
            "topic_id": topic_id,
            "label": info["label"],
            "emoji": info["emoji"],
            "total_likes": total_likes,
            "post_count": len(post_ids),
            "trending_score": trending_score,
            "keywords": keywords.get(topic_id, [])[:5],
        })

    rows.sort(key=lambda r: r["trending_score"], reverse=True)
    return rows[:3]


# ── Search helpers ──────────────────────────────────────────────────────────

def search_by_keywords(query: str, top_k: int = 20) -> pd.DataFrame:
    searcher = build_searcher()
    if isinstance(searcher, InMemorySearcher):
        model = build_embedding_model()
        embedding = model.encode(query, convert_to_numpy=True)
        results = searcher.search_similar_documents(embedding, top_k=top_k)
    else:
        results = searcher.search_similar_documents(query, top_k=top_k)
    return _results_to_df(results)


def search_by_topic(topic_id: int, top_k: int = 20) -> pd.DataFrame:
    searcher = build_searcher()
    if _has_localized():
        emb = np.array(build_localized_embeddings()[topic_id], dtype=np.float32)
    else:
        emb = np.array(build_naive_embeddings()[topic_id], dtype=np.float32)
    results = searcher.search_similar_documents(emb, top_k=top_k)
    return _results_to_df(results)


def _results_to_df(results: list[dict]) -> pd.DataFrame:
    posts_by_id = load_posts_by_id()
    has_topics = _has_topics()
    labels = load_topic_labels() if has_topics else {}
    assignments = load_assignments() if has_topics else pd.DataFrame()
    id_to_label = {tid: f"{v['emoji']} {v['label']}" for tid, v in labels.items()}

    rows = []
    for r in results:
        pid = r.get("post_id", "")
        post = posts_by_id.get(pid, {})
        text = r.get("post_text", "").strip()

        # LO5: show image caption for image-only posts
        caption = post.get("image_caption") or ""
        display_text = text if text else f"[image] {caption}" if caption else "[image — no caption yet]"
        image_url = post.get("image_url")

        topic_label = ""
        if has_topics and not assignments.empty:
            row = assignments.loc[assignments["post_id"] == pid, "topic_id"]
            if len(row):
                tid = int(row.values[0])
                topic_label = id_to_label.get(tid, f"Topic {tid}")

        entry = {
            "score": round(r.get("score", 0), 3),
            "post": display_text,
            "topic": topic_label,
        }
        if image_url:
            entry["image"] = image_url
        rows.append(entry)

    return pd.DataFrame(rows)


# ── Session state ───────────────────────────────────────────────────────────

if "active_topic" not in st.session_state:
    st.session_state.active_topic = None
if "keyword_query" not in st.session_state:
    st.session_state.keyword_query = ""


# ── Detect current workshop stage ──────────────────────────────────────────

HAS_EMBEDDINGS = _has_embeddings()
HAS_TOPICS = _has_topics()
HAS_LOCALIZED = _has_localized()
HAS_CAPTIONS = _has_image_captions()

if HAS_LOCALIZED:
    lo_stage, search_mode = "LO4 complete", "semantic · localized embeddings"
elif HAS_TOPICS:
    lo_stage, search_mode = "LO3 complete", "semantic · topic-centroid embeddings"
elif HAS_EMBEDDINGS:
    lo_stage, search_mode = "LO2 complete", "semantic search"
else:
    lo_stage, search_mode = "LO1", "lexical (keyword match)"


# ── Layout ──────────────────────────────────────────────────────────────────

st.title("Topic Vector Search")

# Small status badge showing current workshop stage
st.markdown(
    f'<p class="mode-badge">🔧 {lo_stage} &nbsp;·&nbsp; search: {search_mode}'
    + (" &nbsp;·&nbsp; 🖼 image captions active" if HAS_CAPTIONS else "")
    + "</p>",
    unsafe_allow_html=True,
)

# — Search bar —
query_input = st.text_input(
    label="search",
    placeholder="Search posts by keywords…",
    value=st.session_state.keyword_query,
    label_visibility="collapsed",
)
if query_input != st.session_state.keyword_query:
    st.session_state.keyword_query = query_input
    st.session_state.active_topic = None

st.markdown("---")

# — Trending topics —
st.subheader("Trending")

if not HAS_TOPICS:
    st.info(
        "Trending topics will appear here after completing **Learning Objective 3** "
        "(Build a Topic Model Pipeline)."
    )
else:
    trending = compute_trending()
    cols = st.columns(3)
    for col, topic in zip(cols, trending):
        with col:
            kw_str = " · ".join(topic["keywords"])
            st.markdown(f"""
<div class="topic-card">
  <div class="topic-emoji">{topic['emoji']}</div>
  <div class="topic-label">{topic['label']}</div>
  <div class="topic-stat">❤️ {topic['total_likes']:,} likes &nbsp;·&nbsp; {topic['post_count']} posts</div>
  <div class="topic-kw">{kw_str}</div>
</div>""", unsafe_allow_html=True)
            if st.button("Search this topic", key=f"btn_{topic['topic_id']}"):
                st.session_state.active_topic = topic["topic_id"]
                st.session_state.keyword_query = ""
                st.rerun()

st.markdown("---")

# — Results —
col_cfg = {
    "score": st.column_config.NumberColumn("Score", format="%.3f", width="small"),
    "post":  st.column_config.TextColumn("Post", width="large"),
    "topic": st.column_config.TextColumn("Topic", width="medium"),
}

if st.session_state.active_topic is not None:
    topic_id = st.session_state.active_topic
    info = load_topic_labels().get(topic_id, {})
    label = f"{info.get('emoji', '')} {info.get('label', f'Topic {topic_id}')}"
    emb_note = "localized embedding" if HAS_LOCALIZED else "topic-centroid embedding"
    st.subheader(f"Results for {label}")
    st.caption(f"Searched using {emb_note}")
    with st.spinner("Searching…"):
        df = search_by_topic(topic_id)
    if "image" in df.columns:
        col_cfg["image"] = st.column_config.ImageColumn("Image", width="small")
    st.dataframe(df, use_container_width=True, column_config=col_cfg, hide_index=True)

elif st.session_state.keyword_query.strip():
    query = st.session_state.keyword_query.strip()
    st.subheader(f'Results for "{query}"')
    st.caption(f"Search mode: {search_mode}")
    with st.spinner("Searching…"):
        df = search_by_keywords(query)
    if "image" in df.columns:
        col_cfg["image"] = st.column_config.ImageColumn("Image", width="small")
    st.dataframe(df, use_container_width=True, column_config=col_cfg, hide_index=True)

else:
    st.markdown(
        "<p style='color:#999;text-align:center;padding:2rem 0'>"
        "Enter keywords above or click a trending topic to search."
        "</p>",
        unsafe_allow_html=True,
    )
