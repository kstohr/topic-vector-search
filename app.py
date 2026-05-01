"""
Topic Vector Search — Workshop Demo
Run with: uv run streamlit run app.py

Update source code in src/ to see changes reflected in the app. All data is
cached for fast reloads.

To see changes, edit files in src/ and save. The app will automatically reload
with the new code and data. The app uses Streamlit's caching to speed up
reloads, so only changes to the source code or data files will trigger updates.
You can also clear the cache if needed.
"""

import base64
import hashlib
import html
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.config import ELASTICSEARCH_URL
from src.evaluation import (
    DEFAULT_EVAL_K,
    DEFAULT_PRECISION_K,
    DEFAULT_RECALL_K,
    SearchResultRowsArgs,
    TopicEvalArgs,
    build_search_result_rows,
    evaluate_topics,
)
from src.search import (
    _SEARCHER_LABELS,
    TEXT_SEARCH_ENGINES,
    TOP_K_DEFAULT,
    TOPIC_SEARCH_ENGINES,
    TextSearchArgs,
    TopicSearchArgs,
    get_searcher,
    get_searcher_label,
    get_topic_searcher,
    run_search_by_text,
    run_search_by_topic,
)
from src.topic_ranking import TopicRankingArgs, TrendingTopic, rank_topics

OUTPUT = Path(__file__).parent / "output"
REPO = Path(__file__).parent

st.set_page_config(page_title="Topic Vector Search", layout="wide")

st.markdown(
    """
<style>
.button_style {
    background-color: #8B6FD4;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.9rem;
    transition: background-color 0.2s ease;
}
/* ── Trending topic cards ─────────────────────────────── */
.topic-card {
    border: 1px solid #e0e0e0; border-radius: 10px;
    padding: 16px; text-align: center;
}
.topic-label  { font-weight: 600; margin: 6px 0 4px; }
.topic-stat   { color: #666; font-size: 0.85rem; }
.topic-kw     { color: #999; font-size: 0.75rem; margin-top: 4px; }

/* ── Social feed cards ────────────────────────────────── */
.feed-card {
    border: 1px solid #e4e4e4;
    border-radius: 12px;
    padding: 18px 20px 12px;
    margin-bottom: 14px;
    background: #fff;
}
.feed-header {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
}
.feed-avatar {
    width: 42px; height: 42px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.95rem;
    color: #fff;
    flex-shrink: 0;
    margin-right: 10px;
    letter-spacing: 0.5px;
}
.feed-author { font-weight: 600; font-size: 0.9rem; line-height: 1.2; }
.feed-time   { margin-left: auto; color: #999; font-size: 0.82rem; white-space: nowrap; }
.feed-topic-tag {
    display: inline-block;
    background: #eef2ff; color: #4a6fa5;
    border-radius: 20px; padding: 2px 10px;
    font-size: 0.72rem; margin-bottom: 8px;
}
.feed-text {
    font-size: 0.92rem; line-height: 1.55;
    color: #222; margin-bottom: 10px;
    white-space: pre-wrap; word-break: break-word;
}
.feed-image {
    width: 100%; border-radius: 8px;
    margin-bottom: 10px; max-height: 380px;
    object-fit: cover; display: block;
}
.feed-footer {
    display: flex; align-items: center; gap: 18px;
    padding-top: 8px; border-top: 1px solid #f0f0f0;
    color: #777; font-size: 0.85rem;
}
.feed-footer .action { display: flex; align-items: center; gap: 4px; }
.feed-score {
    margin-left: auto; font-size: 0.72rem;
    color: #bbb; white-space: nowrap;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Startup: index posts into Elasticsearch ────────────────────────────────


@st.cache_resource
def _startup_index_posts() -> str:
    """Index posts from sample_posts.json into Elasticsearch on first run."""
    from elasticsearch import Elasticsearch

    from src.es_index import INDEX_NAME, create_index

    try:
        client = Elasticsearch(ELASTICSEARCH_URL)
        client.info()
    except Exception:
        return "Elasticsearch unavailable — skipping startup indexing."

    create_index(client)

    if client.count(index=INDEX_NAME).get("count", 0) > 0:
        return f"Index '{INDEX_NAME}' already populated."

    with open(REPO / "sample_posts.json") as f:
        posts = json.load(f)

    for post in posts:
        doc = {k: v for k, v in post.items() if k != "doc_embedding" or v}
        client.index(index=INDEX_NAME, id=post["post_id"], body=doc)

    return f"Indexed {len(posts)} posts into '{INDEX_NAME}'."


_startup_index_posts()


# ── Data loading (all cached) ───────────────────────────────────────────────


@st.cache_data
def load_raw_posts() -> list[dict]:
    with open(REPO / "sample_posts.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_posts_by_id() -> dict[str, dict]:
    return {p["post_id"]: p for p in load_raw_posts()}


@st.cache_data
def load_doc_index() -> list[dict]:
    path = OUTPUT / "processed_posts.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return list(json.load(f).values())
    return load_raw_posts()


# Displayed in evaluation view; not used by search functions themselves.
@st.cache_data
def load_assignments() -> pd.DataFrame:
    path = OUTPUT / "topic_assignments.csv"
    if not path.exists():
        return pd.DataFrame(columns=["post_id", "topic_id"])
    return pd.read_csv(path, encoding="utf-8")


@st.cache_data
def load_topic_keywords() -> dict[int, list[str]]:
    path = OUTPUT / "bertopic_model" / "topics.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    reps = data.get("topic_representations", {})
    return {int(k): [w for w, _ in v] for k, v in reps.items() if int(k) != -1}


@st.cache_data
def load_topic_labels() -> dict[int, dict]:
    path = OUTPUT / "topic_labels.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


@st.cache_resource
def build_searcher(engine: str):
    return get_searcher(load_doc_index(), engine)


@st.cache_resource
def build_topic_searcher(engine: str):
    return get_topic_searcher(load_doc_index(), engine)


@st.cache_data
def load_topic_keyword_embeddings() -> dict[int, list[float]]:
    path = OUTPUT / "topic_keyword_embeddings.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


@st.cache_data
def load_topic_embeddings() -> dict[int, list[float]]:
    path = OUTPUT / "topic_embeddings.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


@st.cache_data
def get_trending() -> list[TrendingTopic]:
    return rank_topics(
        TopicRankingArgs(
            assignments=load_assignments(),
            posts_by_id=load_posts_by_id(),
            labels=load_topic_labels(),
            keywords=load_topic_keywords(),
        )
    )


# ── Search helpers ──────────────────────────────────────────────────────────


def _search_by_text(query: str, top_k: int) -> list[dict]:
    """Text search from the search bar. Works with all four searchers in get_searcher()."""
    return run_search_by_text(
        TextSearchArgs(query=query, searcher=build_searcher(text_engine), top_k=top_k)
    )


def _topic_embeddings() -> dict[int, list[float]]:
    if st.session_state.use_topic_keyword_embeddings:
        return load_topic_keyword_embeddings()
    return load_topic_embeddings()


def _embedding_strategy_label() -> str:
    """Human-readable label for the current topic embedding strategy."""
    return (
        "Topic keyword embedding"
        if st.session_state.use_topic_keyword_embeddings
        else "Topic embedding"
    )


def _search_by_topic(topic_id: int, top_k: int) -> list[dict]:
    """Topic embedding search. Always uses get_topic_searcher() — a semantic searcher."""
    emb = np.array(_topic_embeddings()[topic_id], dtype=np.float32)
    return run_search_by_topic(
        TopicSearchArgs(embedding=emb, searcher=build_topic_searcher(topic_engine), top_k=top_k)
    )


# ── Image helper ────────────────────────────────────────────────────────────


@st.cache_data
def _image_uri(image_url: str) -> str:
    """Convert a local assets/ path to a base64 data URI; pass through http URLs."""
    if not image_url or image_url.startswith("http"):
        return image_url or ""
    img_path = REPO / image_url
    if not img_path.exists():
        return image_url
    suffix = str(img_path).lower()
    if suffix.endswith((".jpg", ".jpeg")):
        mime = "image/jpeg"
    elif suffix.endswith(".gif"):
        mime = "image/gif"
    elif suffix.endswith(".webp"):
        mime = "image/webp"
    else:
        mime = "image/png"
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


# ── Feed rendering ──────────────────────────────────────────────────────────

_AVATAR_COLORS = [
    "#5B8DEF",
    "#8B6FD4",
    "#E8685A",
    "#3CB47A",
    "#E8984A",
    "#4EC5C1",
    "#D46F96",
    "#7AAB58",
]


def _avatar_color(name: str) -> str:
    idx = int(hashlib.md5(name.encode()).hexdigest(), 16) % len(_AVATAR_COLORS)
    return _AVATAR_COLORS[idx]


def _initials(name: str) -> str:
    parts = name.split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return name[:2].upper() if name else "??"


def _rel_time(dt_str: str) -> str:
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        days = (datetime.now(UTC) - dt).days
        if days < 1:
            return "today"
        if days < 7:
            return f"{days}d"
        if days < 30:
            return f"{days // 7}w"
        if days < 365:
            return f"{days // 30}mo"
        return f"{days // 365}y"
    except Exception:
        return ""


def render_feed(results: list[dict]) -> None:
    posts_by_id = load_posts_by_id()

    for r in results:
        pid = r.get("post_id", "")
        post = posts_by_id.get(pid, {})
        author = post.get("post_author", "Unknown")
        created_at = post.get("created_at", "")
        likes = post.get("likes", 0)
        post_text = r.get("post_text", "").strip()
        image_url = post.get("image_url", "")
        score = r.get("score", 0)

        # Avatar
        color = _avatar_color(author)
        initials = _initials(author)
        rel_time = _rel_time(created_at)

        # Text
        text_html = f'<div class="feed-text">{html.escape(post_text)}</div>' if post_text else ""

        # Image
        image_html = ""
        if image_url:
            uri = _image_uri(image_url)
            if uri:
                image_html = f'<img class="feed-image" src="{uri}" alt="post image">'

        card = f"""
<div class="feed-card">
  <div class="feed-header">
    <div class="feed-avatar" style="background:{color}">{html.escape(initials)}</div>
    <div>
      <div class="feed-author">{html.escape(author)}</div>
    </div>
    <div class="feed-time">{html.escape(rel_time)}</div>
  </div>
  {text_html}
  {image_html}
  <div class="feed-footer">
    <span class="action">♡ {likes:,}</span>
    <span class="action">○</span>
    <span class="action">↗</span>
    <span class="feed-score">score {score:.3f}</span>
  </div>
</div>"""
        st.markdown(card, unsafe_allow_html=True)


def render_results_eval(results: list[dict]) -> None:
    rows = build_search_result_rows(
        SearchResultRowsArgs(
            results=results,
            posts_by_id=load_posts_by_id(),
            assignments=load_assignments(),
            labels=load_topic_labels(),
        )
    )
    dicts = []
    for row in rows:
        d = row.model_dump()
        if d["image_url"]:
            d["image"] = _image_uri(d.pop("image_url"))
        else:
            d.pop("image_url")
        dicts.append(d)

    df = pd.DataFrame(dicts)
    col_cfg = {
        "score": st.column_config.NumberColumn(
            "Score",
            format="%.3f",
            width="small",
            help="Search score, normalized to [0, 1] for easier comparison across engines.",
        ),
        "post": st.column_config.TextColumn(
            "Post", width="large", help="The text content of the post."
        ),
        "topic": st.column_config.TextColumn(
            "Topic", width="medium", help="The topic label assigned by BERTopic, if available."
        ),
    }
    if "image" in df.columns:
        col_cfg["image"] = st.column_config.ImageColumn(
            "Image", width="small", help="The image associated with the post, if available."
        )
    st.dataframe(df, width="stretch", column_config=col_cfg, hide_index=True)


# ── Session state ───────────────────────────────────────────────────────────

if "active_topic" not in st.session_state:
    st.session_state.active_topic = None
if "keyword_query" not in st.session_state:
    st.session_state.keyword_query = ""
if "search_input" not in st.session_state:
    st.session_state.search_input = ""
if "_clear_search" not in st.session_state:
    st.session_state._clear_search = False
if "eval_view" not in st.session_state:
    st.session_state.eval_view = False
if "show_topic_eval" not in st.session_state:
    st.session_state.show_topic_eval = False
if "use_topic_keyword_embeddings" not in st.session_state:
    st.session_state.use_topic_keyword_embeddings = False
if "eval_selected_topic_id" not in st.session_state:
    st.session_state.eval_selected_topic_id = None

# Clear the search widget before it is instantiated (cannot be set after render)
if st.session_state._clear_search:
    st.session_state.search_input = ""
    st.session_state._clear_search = False


# ── Layout ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Demo App Controls")

    st.caption("Clear the cache to pick up source code or data changes.")
    if st.button("Clear cache", width="stretch"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.header("Search Controls")
    st.caption(
        "Select search engine for the search bar and topic search. Changes apply on next search."
    )

    text_engine = st.radio(
        "Search bar engine",
        options=TEXT_SEARCH_ENGINES,
        format_func=lambda k: _SEARCHER_LABELS[k],
    )
    topic_engine = st.radio(
        "Topic search engine",
        options=TOPIC_SEARCH_ENGINES,
        format_func=lambda k: _SEARCHER_LABELS[k],
    )

    st.header("Evaluation")

    def _on_eval_view_toggle():
        st.session_state.eval_view = not st.session_state.eval_view

    def _on_topic_eval_toggle():
        st.session_state.show_topic_eval = not st.session_state.show_topic_eval
        st.session_state.active_topic = None
        st.session_state.eval_selected_topic_id = None
        st.session_state.keyword_query = ""
        st.session_state._clear_search = True

    st.toggle(
        "Evaluate search results",
        value=st.session_state.eval_view,
        key="eval_view_sidebar",
        on_change=_on_eval_view_toggle,
    )

    st.toggle(
        "Evaluate topics", value=st.session_state.show_topic_eval, on_change=_on_topic_eval_toggle
    )

    st.toggle(
        "Use topic keyword embeddings",
        key="use_topic_keyword_embeddings",
        help=(
            "On: topic keyword embeddings from top keywords (topic_keyword_embeddings.json). "
            "Off: topic embeddings — mean of assigned doc embeddings (topic_embeddings.json)."
        ),
    )

st.title("Topic Vector Search")


# — Search bar —
def _on_search_change():
    st.session_state.keyword_query = st.session_state.search_input
    st.session_state.active_topic = None
    st.session_state.show_topic_eval = False


st.text_input(
    label="search",
    placeholder="Search posts by keywords…",
    key="search_input",
    on_change=_on_search_change,
    label_visibility="collapsed",
)

st.markdown("---")

# — Trending topics —
st.subheader("Trending")

trending = get_trending()
cols = st.columns(3)
for col, topic in zip(cols, trending, strict=False):
    with col:
        kw_str = " · ".join(topic.keywords)
        st.markdown(
            f"""
<div class="topic-card">
  <div class="topic-label">{topic.label}</div>
  <div class="topic-stat">❤️ {topic.total_likes:,} likes · {topic.post_count} posts</div>
  <div class="topic-kw">{kw_str}</div>
</div>""",
            unsafe_allow_html=True,
        )
        if st.button("Search this topic", key=f"btn_{topic.topic_id}"):
            st.session_state.active_topic = topic.topic_id
            st.session_state.keyword_query = ""
            st.session_state._clear_search = True
            st.session_state.show_topic_eval = False
            st.rerun()

if st.session_state.show_topic_eval:
    with st.spinner("Evaluating topics…"):
        labels = load_topic_labels()
        eval_df: pd.DataFrame = evaluate_topics(
            TopicEvalArgs(
                corpus_size=len(load_posts_by_id()),
                labels=labels,
                assignments=load_assignments(),
                topic_embeddings=_topic_embeddings(),
                searcher=build_topic_searcher(topic_engine),
                keywords=load_topic_keywords(),
            )
        )

    topic_ids = sorted(labels.keys())

    selection = st.dataframe(
        eval_df,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "topic": st.column_config.TextColumn(
                "Topic", width="medium", help="The topic label assigned by BERTopic, if available."
            ),
            "keywords": st.column_config.TextColumn(
                "Keywords",
                width="large",
                help="Top keywords by c-TF-IDF score, computed during BERTopic training.",
            ),
            "assigned_posts": st.column_config.NumberColumn(
                "Assigned",
                width="small",
                help="Count of posts assigned to this topic based on BERTopic clustering.",
            ),
            "avg_search_score": st.column_config.NumberColumn(
                "Avg Search Score",
                format="%.3f",
                width="small",
                help="Average semantic search score across the evaluation search results.",
            ),
            "precision_at_k": st.column_config.NumberColumn(
                f"Precision@{DEFAULT_PRECISION_K}",
                format="%.3f",
                width="small",
                help=(
                    f"Quality of the top {DEFAULT_PRECISION_K} search results: "
                    "precision_hits / precision_k."
                ),
            ),
            "recall_at_k": st.column_config.NumberColumn(
                f"Recall@{DEFAULT_RECALL_K}",
                format="%.3f",
                width="small",
                help=(
                    f"Coverage of the topic within the top {DEFAULT_RECALL_K} search results: "
                    "recall_hits / assigned topic posts."
                ),
            ),
            "baseline": st.column_config.NumberColumn(
                "Random Baseline",
                format="%.3f",
                width="small",
                help="Expected precision from randomly selecting posts from the full corpus.",
            ),
            "num_posts_assigned_to_topic": st.column_config.NumberColumn(
                "Topic Posts",
                width="small",
                help="Posts assigned to this topic by BERTopic.",
            ),
            "num_retrieved_by_search": st.column_config.NumberColumn(
                "Retrieved",
                width="small",
                help=f"Results returned for evaluation (up to eval_k={DEFAULT_EVAL_K}).",
            ),
            "precision_hits": st.column_config.NumberColumn(
                "Precision Hits",
                width="small",
                help=f"Assigned topic posts found in the top {DEFAULT_PRECISION_K} results.",
            ),
            "recall_hits": st.column_config.NumberColumn(
                "Recall Hits",
                width="small",
                help=f"Assigned topic posts found in the top {DEFAULT_RECALL_K} results.",
            ),
        },
    )

    selected_rows = selection.selection.rows if selection else []
    if selected_rows:
        st.session_state.eval_selected_topic_id = topic_ids[selected_rows[0]]

    active_eval_topic = st.session_state.eval_selected_topic_id
    if active_eval_topic is not None:
        info = labels[active_eval_topic]
        results = _search_by_topic(active_eval_topic, top_k=DEFAULT_EVAL_K)

        st.subheader(f"Results for {info['label']}")
        engine_label = get_searcher_label(build_topic_searcher(topic_engine))
        st.caption(f"Search engine: {engine_label} · Embedding: {_embedding_strategy_label()}")

        if st.session_state.eval_view:
            render_results_eval(results)
        else:
            render_feed(results)

st.markdown("---")

# — Results header + view toggle —
if not st.session_state.show_topic_eval and (
    st.session_state.active_topic is not None or st.session_state.keyword_query.strip()
):
    topic_id = st.session_state.active_topic
    query = st.session_state.keyword_query.strip()

    # Build header text and run search
    if topic_id is not None:
        info = load_topic_labels().get(topic_id, {})
        label = info.get("label", f"Topic {topic_id}")
        header_text = f"Results for {label}"
        with st.spinner("Searching…"):
            results = _search_by_topic(topic_id, top_k=TOP_K_DEFAULT)
    else:
        header_text = f'Results for "{query}"'
        with st.spinner("Searching…"):
            results = _search_by_text(query, top_k=TOP_K_DEFAULT)

    st.subheader(header_text)
    if topic_id is not None:
        engine_label = get_searcher_label(build_topic_searcher(topic_engine))
        st.caption(f"Search engine: {engine_label} · Embedding: {_embedding_strategy_label()}")
    else:
        st.caption(f"Search engine: {get_searcher_label(build_searcher(text_engine))}")

    if st.session_state.eval_view:
        render_results_eval(results)
    else:
        render_feed(results)

else:
    st.markdown(
        "<p style='color:#999;text-align:center;padding:2rem 0'>"
        "Enter keywords above or click a trending topic to search."
        "</p>",
        unsafe_allow_html=True,
    )
