# Thinking of Topic Modeling as Search
Use vector search to search for documents related to topics derived from a topic
model.

# Requirements
- python 3.12
- [poetry]() (package manager)
- [Docker]()
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Open Search Python SDK]()

# Installation
After cloning the repository and change directory to the project root:

1. Run Docker Compose
This will install a local version of the Open Search database.
```
docker compose up
```
2. Install Poetry
 ```
 pipx install poetry
 poetry init
 poetry shell
 ```

# Comments
- This demonstration depends on OpenSearch, which is a forked version of ElasticSearch,
  and is now supported by AWS's managed service rather than ElasticSearch.
  However, this functionality also exists on
  [ElasticSearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html).
  The code to run the search is slightly different and you will need to adapt
  this example.

  # Reference
  Open Search Vector Search:
  [https://opensearch.org/docs/latest/search-plugins/vector-search/](https://opensearch.org/docs/latest/search-plugins/vector-search/)
  BERTopic: