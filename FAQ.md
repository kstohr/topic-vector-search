#FAQ 

Common questions about embeddings, topic modeling and search. 

Q: Traditional "full text" searches, including standard searches using Elasticsearch work
well. When should you consider using "semantic" search instead? 

Most full text search engines do more than a simple "keyword" search.
Elasticsearch is a search engine built on Apache Lucene. It works well in most
use cases. It's full text search does handle tokenizing, lower case filters and removing
stop words, as well as lemming, stemming and synonyms, if defined. Postgres
search extensions can do similar full text searches. However, full text search
can't infer semantic connections between documents that may contain similar concepts
but use different terms. If you need more robust searches on similar concepts, 
you should consider using semantic search instead. 

Q: What is an embedding exactly? How is that related to the term "embedding
space"? What is meant by the term "latent features"? 

Q: How does an embedding model such as "all-MiniLM-L6-v2" differ from an LLM
(Large Language Model)? 

Q: How are HDBSCAN and other clustering algorithms different than KNN using
cosing similarity. They see to do the same thing? 

