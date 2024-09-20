# Thinking of Topic Modeling as Search
Use vector search to search for documents related to topics derived from a topic
model.

# Presentation
[Thinking of Topic Modeling as Search](https://docs.google.com/presentation/d/1UfaDLzG9WvTeP8I64za-ycwnst8bXMzx/edit?usp=sharing&ouid=105992325138979778362&rtpof=true&sd=true)


# Requirements
- python 3.12
- [poetry]() (package manager)
- [Docker]()
- [Docker Compose](https://docs.docker.com/compose/install/)

# Installation
After cloning the repository and change directory to the project root:

1. Run Docker Compose
This will install a local version of the Open Search database.
```
docker compose up
```
2. Install and launch Poetry environment
 ```
 pipx install poetry
 poetry install
 poetry shell
 ```
3. Create a .env file with OPENAI credentials (or adapt to the service of your choice)
```
OPENAI_API_KEY="sk-some-key"
OPENAI_ORGANIZATION="org-some-org-or-none"
OPENAI_PROJECT="proj_some-project-or-none"
```
# Comments
This demonstration depends on OpenSearch, which is a forked version of
ElasticSearch, and is now supported by AWS's managed service rather than
ElasticSearch. However, this functionality also exists on
[ElasticSearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html).
The code to run the search is slightly different and you will need to adapt this example.

# References and Credits
Thanks go to Maarten Grootendorst for his work and excellent documentation in
[BERTopic](https://github.com/MaartenGr/BERTopic) as well as colleagues at
Unified and coding partner, [Ray 'Urgent'McLendon](https://www.linkedin.com/in/raymclendon/) for his interest and input.

Open Search Vector Search:  
[https://opensearch.org/docs/latest/search-plugins/vector-search/](https://opensearch.org/docs/latest/search-plugins/vector-search/)

Text Embeddings:
[https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/](https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/)

BERTopic - Package for topic modeling by Maarten Grootendorst  
[https://github.com/MaartenGr/BERTopic](https://github.com/MaartenGr/BERTopic)

Comparing Clustering Algorithms (HDBSCAN)  
[https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html](https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html)

c-TF-IDF  
[https://www.maartengrootendorst.com/blog/ctfidf/](https://www.maartengrootendorst.com/blog/ctfidf/)

Vector Search with Text Search  
[https://towardsdatascience.com/text-search-vs-vector-search-better-together-3bd48eb6132a](https://towardsdatascience.com/text-search-vs-vector-search-better-together-3bd48eb6132a)  

[https://machine-mind-ml.medium.com/enhancing-llm-performance-with-vector-search-and-vector-databases-1f20eb1cc650](https://machine-mind-ml.medium.com/enhancing-llm-performance-with-vector-search-and-vector-databases-1f20eb1cc650)

