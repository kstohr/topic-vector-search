# FAQ

Common questions about embeddings, topic modeling, and search.

---

**Q: Traditional "full text" searches, including standard searches using
Elasticsearch, work well. When should you consider using "semantic" search
instead?**

Most full-text search engines do more than a simple keyword match. Elasticsearch
(built on Apache Lucene) does tokenization, lowercasing, stop-word removal,
stemming/lemmatization, and configurable synonym expansion. Postgres has similar
extensions. For most production workloads on well-curated content where users type
queries that overlap with the document vocabulary, BM25-style full-text search is
fast, cheap, explainable, and good enough.

Reach for **semantic search** when one or more of these is true:

- **Query and document use different words for the same idea.** Users search
  *"how do I cancel my subscription?"* but your help articles say *"unsubscribe"* or
  *"close account"*. BM25 won't bridge those.
- **Paraphrases and natural-language queries.** Long descriptive queries (*"a small
  dog that's good with kids"*) match the *concept* but rarely the exact tokens of
  any one document.
- **Cross-language or mixed-language content.** Multilingual embedding models put
  *"cat"* and *"chat"* near each other; lexical search treats them as unrelated.
- **Topical / conceptual retrieval.** This workshop is the canonical example —
  given a topic embedding (a centroid in the 384-d space), what documents are
  conceptually closest to it? There's no "query string" to BM25 against.
- **Noisy or short text.** Tweets, chat messages, log lines — texts so short that
  BM25's term-frequency signal is mostly missing.

In practice, **hybrid search** (BM25 + vector, scored together) usually beats either
alone. Lexical search anchors on exact-match precision (product names, IDs, code
identifiers); semantic search adds recall on synonyms and paraphrases. Elasticsearch
8.x supports running both in one query and combining the scores.

A good heuristic: if your users would be happy with a "Ctrl-F"-like experience,
BM25 is probably enough. If they'd be happier with a "find me things *like* this"
experience, you want semantic search.

---

**Q: What is an embedding exactly? How is that related to the term "embedding
space"? What is meant by "latent features"?**

An **embedding** is a fixed-length vector of floats produced by a model. With
`all-MiniLM-L6-v2`, every input — single word, sentence, or paragraph — becomes a
384-dimensional vector. The model is trained so that semantically similar inputs
end up at nearby points and dissimilar inputs end up far apart, measured by cosine
distance. So *"a tabby cat napping"* and *"the kitten is asleep"* land near each
other; *"the stock market crashed"* lands somewhere else.

The **embedding space** is just the geometric coordinate system those vectors live
in. For `all-MiniLM-L6-v2` it's a 384-d space (for `text-embedding-3-large` it's
3072-d). All the geometric intuition you'd have about 3-d still applies — distance,
direction, clustering — it just happens in higher dimensions. *Semantic search* is
literally "find the nearest neighbours of the query vector in this space".

**Latent features** are the *individual dimensions* of the embedding. They're
"latent" — hidden — because the model learned them from training data and no human
labelled what each dimension means. One dimension might roughly correlate with
"is about animals", another with "is a question", another with "negative sentiment"
— but in practice each dimension encodes a noisy mixture of many such ideas. The
useful thing isn't any single latent feature but the *whole vector*: cosine
similarity over all 384 dimensions captures meaning even when no individual
dimension does.

---

**Q: How does an embedding model such as "all-MiniLM-L6-v2" differ from an LLM
(Large Language Model)?**

They share the same underlying architecture (the **transformer**), but they're
trained for different jobs and used in different ways.

| | Embedding model (MiniLM) | LLM (GPT-4, Llama, Qwen) |
|---|---|---|
| Output | One fixed-size vector | A sequence of generated tokens |
| Trained to... | Place similar texts near each other | Predict the next token given context |
| Size | ~22M parameters | Billions to trillions |
| Speed | Milliseconds per input on CPU | Seconds per response, often needs GPU |
| You ask it... | "Encode this string" | "Continue this prompt" |
| Used for | Search, clustering, classification | Generation, chat, summarization, labelling |

A useful mental model: an embedding model is a **measuring instrument**. You hand
it a string, it hands back coordinates. It does not "understand" or "answer" —
it places.

An LLM is a **generator**. You hand it a prompt, it produces text. It can answer,
summarize, translate, label, etc.

In this workshop both show up: `all-MiniLM-L6-v2` produces the document and topic
embeddings that power search; an LLM (Ollama or OpenAI) is asked to generate a
short label for each topic given its top keywords. They're complementary, not
competitive.

(Trivia: many production embedding models started life as the *encoder half* of an
LLM-style architecture and were then fine-tuned with a contrastive objective so
that paraphrases land close together in vector space.)

---

**Q: How are HDBSCAN and other clustering algorithms different than KNN using
cosine similarity? They seem to do the same thing.**

They look similar because both operate on vector distances, but they're answering
different questions.

**KNN** is a *retrieval* / *lookup* operation. You hand it:
- a *query point*, and
- a number *K*.

It returns the *K* points in your dataset that are nearest to the query. There is
no notion of "groups" — every query gets its own personalized top-K. KNN is what
powers semantic *search*: at query time, find the K nearest documents to the
query embedding.

**HDBSCAN** (and k-means, DBSCAN, agglomerative clustering, etc.) is a
*partitioning* operation. You hand it:
- the *whole dataset*.

It returns a label per point indicating which cluster it belongs to (or that it's
an outlier). No query, no K. It's answering "what natural groupings exist in this
data?", not "what's near this specific point?".

A few more concrete differences:

- **Direction of the question.** KNN: "Given this query, what's near it?"
  Clustering: "Given all the data, where are the dense regions?"
- **Specifying K.** KNN's K is the *number of results* you want back. HDBSCAN
  doesn't take a K — the data determines how many clusters there are. (k-means
  *does* require K, which is one reason BERTopic prefers HDBSCAN.)
- **Outliers.** HDBSCAN can label a point as `-1` (no cluster). KNN always returns
  K results, even if they're all far away.
- **Output shape.** KNN gives you a *ranked list*. Clustering gives you a *partition*
  of the input.

In this workshop both run, on the same embeddings, in different stages:

1. BERTopic uses HDBSCAN (after UMAP) to *partition* documents into topics —
   this is offline, training-time work.
2. The demo app uses KNN over cosine similarity to *retrieve* documents at query
   time, given either a user's text query or a topic centroid embedding.

So: clustering builds the topics; KNN searches over them. They're complementary
parts of the same pipeline.

---

**Q: What is BERT? What is BERTopic? What is KeyBERT?**

These three names come up together a lot and it's easy to conflate them. They share
a lineage but solve different problems.

**BERT** (*Bidirectional Encoder Representations from Transformers*, Google, 2018)
is a transformer-encoder model. Given an input text, it produces a contextual
vector for every input token (and a single pooled vector for the whole sequence).
It was a big deal in 2018 because it was the first model to popularize
*pre-train then fine-tune* for language: train once on a huge corpus with a generic
objective, then fine-tune cheaply for any downstream task. Modern sentence
transformers like `all-MiniLM-L6-v2` are descendants of BERT — smaller, distilled,
and specifically fine-tuned to produce one good *sentence-level* vector.

**BERTopic** (Maarten Grootendorst, 2020) is a Python topic-modeling **library**.
It assembles a pipeline of off-the-shelf components:

```
documents → embed (BERT-family) → UMAP → HDBSCAN → CountVectorizer
         → c-TF-IDF → KeyBERT/LLM (label) → topics
```

Each stage is configurable — you can swap the embedding model, the dimensionality
reducer, the clustering algorithm, and the labelling step. BERTopic doesn't *invent*
any one of those algorithms; it's the glue that makes them work well together for
topic modeling. In this repo, BERTopic is what `src/topic_model.py` orchestrates.

**KeyBERT** (also Maarten Grootendorst) is a small **library** for *keyword
extraction*. Given a single document and an embedding model, it:

1. Embeds the whole document into one vector.
2. Generates candidate n-grams from the document.
3. Embeds each candidate.
4. Returns the n-grams whose embeddings have the highest cosine similarity to the
   document embedding.

Result: the n-grams in the document that are most representative of its overall
meaning. BERTopic uses a "KeyBERTInspired" representation model to do the same
thing per *cluster* — picking the n-grams most representative of each topic
centroid, which is what feeds the *localized* topic embeddings explored in
Notebook 5.

**Quick disambiguation:**
- BERT → a *model architecture / family of pre-trained weights*.
- BERTopic → a *library* for end-to-end topic modeling.
- KeyBERT → a *library* for keyword extraction. BERTopic uses it as one component.

You can absolutely use BERTopic without ever caring about the original BERT — and
in fact this workshop uses MiniLM (a much smaller BERT descendant) throughout.
