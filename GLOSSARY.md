# Glossary

Terms used throughout the workshop. Aimed at a software developer who hasn't done
much NLP / IR / ML work before. Listed alphabetically.

---

**all-MiniLM-L6-v2** — The 22M-parameter sentence-transformer used everywhere in this
workshop. Takes any string and returns a 384-dimensional embedding. Small, fast, and
runs comfortably on CPU. See *Sentence transformer*.

**Attention** — The mechanism in a transformer that lets each token "look at" the
other tokens in the input and update its own vector based on what's relevant. The
reason transformers can produce *contextual* embeddings (the word *bank* near
*river* ends up at a different point in the embedding space than *bank* near *loan*).

**Baseline (random) precision** — The precision you'd expect from picking results
uniformly at random. For a topic of size *t* in a corpus of size *N*, it's just
`t / N`. Used as a sanity check: if your search precision isn't well above random,
the search isn't doing much.

**BERT** — *Bidirectional Encoder Representations from Transformers*. A 2018
transformer-encoder model from Google. The "B" in BERTopic and KeyBERT refers to
the family of BERT-derived embedding models used inside them.

**BERTopic** — The topic-modeling library this workshop is built on. A configurable
pipeline: `embed → UMAP (reduce dims) → HDBSCAN (cluster) → CountVectorizer + c-TF-IDF
(rank words per topic) → KeyBERT/LLM (label)`. https://maartengr.github.io/BERTopic/

**BLIP** — *Bootstrapping Language-Image Pre-training*. The vision-language model
used in Notebook 6 to generate text captions for image-only posts so they can be
embedded by the same text encoder.

**BM25** — BM25 (Best Matching 25) is a ranking algorithm used in search. 
 It is the default ranking function in Lucene and Elasticsearch. 
Scores documents by how often the query terms appear, dampened for very
common terms and adjusted for document length (See: Term Frequency × Inverse 
Document Frequency). Token-based — it cannot bridge synonyms. However,
Elsaticsearch and other search engines do allow additional synonym handling, 
BM25 does not handle synonyms by default.

**c-TF-IDF (class-based TF-IDF)** — A BERTopic variant of TF-IDF (Term Frequency
 × Inverse Document Frequency) where each *cluster*
of documents is treated as one "class document". Highlights the n-grams that are
common *inside* a topic but rare *outside* it — i.e. words that distinguish the
topic from the others.

**Centroid** — The geometric centre (mean) of a set of vectors as projected onto
a multi-dimensional space. In BERTopic, the *topic embedding* is the centroid of
the document embeddings assigned to that topic. While it is referred to as the
"centroid" it is important to remember that it is an unweighted representation
of all terms in all documents assigned to the topic as opposed to a "localized"
embedding which is generated from the top-n terms and documents most similar to 
the topic embedding.

**Class document** — In c-TF-IDF, all documents of one class (topic) concatenated
into a single string before vectorizing. Lets you compute "TF-IDF per class".

**Cluster** — A group of points that are near each other in some vector space.
Producing clusters is what HDBSCAN does on the UMAP-reduced embeddings.

**Coherence (topic)** — A measure of whether the words inside a topic actually go
together. High coherence = the top words look like they describe one thing.

**Corpus** — The complete set of documents you're working with. In this workshop,
the `sample_posts.json` (and its noisy variant) is the corpus.

**Cosine similarity** — A similarity score between two vectors, computed as
`dot(a, b) / (|a| * |b|)`. Range `-1` (opposite) → `0` (orthogonal) → `1`
(same direction). Insensitive to vector *magnitude* — it only measures *direction*,
which is exactly what you want for embeddings.

**CountVectorizer** — A scikit-learn class that turns a list of strings into a
sparse matrix of n-gram counts. Used by BERTopic to build the per-topic vocabulary
before c-TF-IDF re-weights it.

**Demojize** — Converting an emoji character (`🐈`) into a text token (`:cat:` or
`cat`). Done in `PostDocument.preprocess_text()` so the embedding model can use the
emoji's meaning.

**Dense vector** — A fixed-length vector where (almost) every dimension carries
information — e.g. `[-0.02, 0.08, ..., 0.05]` over 384 dimensions. Embeddings are
dense vectors. Contrast with *sparse vector*.

**Diversity (topic)** — A measure of how distinct topics are from each other. High
diversity = the heatmap shows low cosine similarity off the diagonal.

**Document** — In this workshop, one social-media post represented as a
`PostDocument`. More generally, the unit of text you embed and retrieve.

**Doc embedding** — The 384-dim vector that represents one document. Stored on
`PostDocument.doc_embedding` and indexed in Elasticsearch.

**Elasticsearch** — The search engine used by the demo app. Stores documents,
supports BM25 keyword search, and supports vector similarity search via `script_score`
queries. The repo also ships an in-memory fallback if Docker isn't running.

**Embedding** — The output of an embedding model: a dense vector representing the
meaning of a piece of text (or image, audio, etc.). Two semantically similar inputs
end up at nearby points.

**Embedding space** — The high-dimensional coordinate system the embeddings live in
(384-d for `all-MiniLM-L6-v2`). "Similar inputs cluster together in the embedding
space" is the property that makes semantic search work.

**Encoder** — A model (or model component) that maps an input to a vector
representation. Sentence transformers are encoders; LLMs are mostly *decoders*
(they generate text). BLIP is technically an encoder-decoder.

**eval-K (retrieval depth)** — The `size` parameter passed to a search query when
*evaluating* a topic. Set large enough to plausibly capture all of a topic's posts.

**Ground truth** — The "correct" answer used as the reference when computing
metrics. Here, the topic assignments BERTopic produced are the ground truth against
which we measure topic-embedding *search* quality.

**HDBSCAN** — *Hierarchical Density-Based Spatial Clustering of Applications with
Noise*. A clustering algorithm that finds dense regions of points and labels
everything else as outliers (label `-1`). Unlike k-means, you don't tell it how
many clusters to find.

**Image captioning** — Generating a text description of an image with a
vision-language model (BLIP here). The caption is then embedded with the same text
encoder, so image-only posts become searchable by text query.

**Index (Elasticsearch)** — The Elasticsearch equivalent of a database table.
Posts are stored in the `post_docs` index.

**Intertopic distance** — Pairwise distance between topic embeddings.
BERTopic's `visualize_topics()` plots these on a 2-D map; topics far apart are
well-separated.

**KeyBERT** — A library that ranks candidate n-grams in a document by cosine
similarity to the *document's* embedding. BERTopic uses a "KeyBERT-Inspired"
representation to pick the top n-grams per topic that best match the topic embedding.

**KNN (k-nearest neighbours)** — Given a query point, return the *K* points in your
dataset closest to it under some distance metric. Semantic search is KNN over
document embeddings using cosine similarity.

**Latent features** — The dimensions of an embedding. They're "latent" because the
model learned them from data — they don't correspond to human-defined attributes
like "is about cats" or "is in English". One latent dimension probably encodes a
mix of many such ideas.

**Lemmatization** — Reducing a word to its dictionary form: *running* → *run*,
*better* → *good*. More linguistic than stemming. Used by some lexical search
engines so that *running* and *runs* match the same query.

**Lexical search** — Search by token match. Document and query are compared as bags
of tokens (BM25, substring match, etc.). Cannot match synonyms or paraphrases.
Contrast with *semantic search*.

**LLM (large language model)** — A large transformer trained to *generate* text
token-by-token (GPT-4, Llama, Qwen). In this workshop, an LLM (Ollama or OpenAI) is
used only to write a one-line topic *label* given the top keywords. Different job
from the embedding model.

**Localized embedding** — An embedding built from the *top KeyBERT keywords*
rather than from the topic embedding. Useful when the topic has
drifted toward the middle of the embedding space because of diffuse documents.

**Mean pooling** — How a sentence transformer collapses a variable-length sequence
of token vectors into a single fixed-size vector: take the per-token hidden vectors,
ignore padding tokens with the attention mask, and average the remaining token
vectors dimension-wise. In other words, it is a *masked mean*, not a blind
`np.mean(...)` over every token slot. That's why a one-word and a 200-word input
both produce a `(384,)` embedding. 

$$
v_{\text{sent}} = \frac{\sum_i m_i h_i}{\sum_i m_i}
$$

where $h_i$ is token vector $i$, and $m_i \in \{0,1\}$ is the attention-mask value.

**min_df / max_df** — `CountVectorizer` parameters. `min_df` drops terms that
appear in fewer than that many documents (kills typos and one-offs). `max_df` drops
terms in more than that fraction of documents (kills very common terms).
Important: In the BERTopic package all documents in a topic are stored in a
single document and passed to CountVectorizer, so these terms are applied across
topics. A term must appear in more than or fewer than n-topics to be included.

**N-gram** — A contiguous run of *n* tokens. Unigrams = single words, bigrams =
two-word phrases, trigrams = three-word phrases. `ngram_range=(1, 3)` keeps all
three. Capturing bigrams/trigrams lets the model see *machine learning* as one
concept instead of two.

**Ollama** — A tool for running LLMs locally (no API key, no internet). The repo
uses it to generate topic labels via `qwen2.5:3b` if it's available.

**Outlier (HDBSCAN `-1`)** — A document HDBSCAN couldn't confidently place in any
cluster. Not "bad" data — just not part of any dense region. They still appear in
search results, which is part of the noise problem in Notebook 5.

**Pre-trained model** — A model whose weights have already been trained on a large
corpus and shipped for reuse. The whole workshop assumes you don't train an
embedding model from scratch — you download `all-MiniLM-L6-v2` and use its weights.

**Precision@K** — Of the top-*K* search results, how many are *relevant*? `hits / K`.
Tracks the *first-impression* quality of search.

**Probability (BERTopic)** — Per-document, per-topic membership confidence (a number
in [0, 1]) computed by HDBSCAN's soft-clustering mode. Stored in
`output/probabilities.json`.

**Recall@K** — Of the relevant documents, how many appear in the top-*K* results?
`hits / total_relevant`. Tracks *coverage*.

**Retrieval** — The act of fetching documents that match a query. Distinct from
*classification* (assigning a label). In this workshop, topic search is retrieval
(rank docs by how on-topic they look), not classification.

**script_score** — An Elasticsearch query type that lets you supply a custom scoring
script. We use it to compute `cosineSimilarity(query_vector, doc_embedding) + 1.0`
over all stored documents.

**Semantic search** — Search by *meaning*. Encode the query and documents into
embeddings; rank documents by cosine similarity to the query embedding. Bridges
synonyms, paraphrases, and even cross-language matches.

**Sentence transformer** — A transformer model fine-tuned to produce one
fixed-length embedding per input string (rather than per-token vectors).
`all-MiniLM-L6-v2` is one. https://www.sbert.net/

**Sparse vector** — A vector that's mostly zero. A bag-of-words representation over
a 30,000-word vocabulary is a 30,000-dim sparse vector with at most a few hundred
non-zeros. Contrast with *dense vector*.

**Stemming** — Crude rule-based truncation of words: *running* → *runn*,
*horses* → *hors*. Faster than lemmatization but dumber.

**Stop words** — Very common words (*the*, *and*, *of*, *is*) that carry little
topical signal. Most lexical search engines and `CountVectorizer` drop them by
default.

**Streamlit** — The Python web framework powering `app.py`. Lets you write a
multi-page interactive app with pure Python — no JS or HTML.

**Tensor** — A multi-dimensional array (the generalisation of a vector or matrix).
The image processor in Notebook 6 returns a `[1, 3, 224, 224]` tensor: 1 image × 3
RGB channels × 224 pixel height × 224 pixel width.

**TF-IDF** — *Term Frequency × Inverse Document Frequency*. A classical scoring
that highlights words frequent in one document but rare across the corpus. The
inspiration for c-TF-IDF (which compares classes/topics instead of individual
documents).

**Token** — The unit a model operates on after splitting input text. For
`all-MiniLM-L6-v2` (and most BERT-family models), tokens are sub-word pieces — the
word *kitten* might be one token, *kittenish* might be split into `kitten` + `##ish`.

**Tokenization** — The step of splitting raw text into tokens. The first thing any
NLP model does to your input.

**top_k** — How many results to return from a search. The "K" in KNN, precision@K,
recall@K — but these aren't the same K (see *eval-K*).

**Transformer** — The neural-network architecture (Vaswani et al., 2017) underlying
nearly every modern language model. Built around attention layers. Both BERT-family
encoders and GPT-family decoders are transformers.

**UMAP** — *Uniform Manifold Approximation and Projection*. A dimensionality-reduction
algorithm that takes high-dimensional embeddings (384-d here) and projects them
down to 2–10 dimensions while preserving local structure. BERTopic uses it before
HDBSCAN because density-based clustering struggles in high dimensions.

**Vector database** — A database optimised for storing many embeddings and
answering KNN queries quickly. Elasticsearch (8.x+), pgvector, Pinecone, Weaviate
are all examples. We use Elasticsearch.

**Vision-language model** — A model trained jointly on images and text, capable of
generating one from the other. BLIP is a small open-weights VLM used here for image
captioning.

**Vocabulary** — The set of n-grams a `CountVectorizer` decides to keep after
applying `min_df` / `max_df` / `stop_words` filters.
