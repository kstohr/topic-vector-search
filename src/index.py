from elasticsearch import Elasticsearch

INDEX_NAME = "post_docs"
INDEX_BODY = {
    "settings": {"number_of_shards": 1},
    "mappings": {
        "properties": {
            "post_id":     {"type": "keyword"},
            "post_author": {"type": "keyword"},
            "created_at":  {"type": "date"},
            "modified_at": {"type": "date"},
            "post_text":   {"type": "text"},
            "doc_embedding": {
                "type":       "dense_vector",
                "dims":       384,
                "index":      True,
                "similarity": "cosine",
            },
        }
    },
}


def get_client() -> Elasticsearch:
    return Elasticsearch("http://localhost:9200")


def create_index(client: Elasticsearch) -> None:
    if not client.indices.exists(index=INDEX_NAME):
        client.indices.create(index=INDEX_NAME, body=INDEX_BODY)
        print(f"Index '{INDEX_NAME}' created successfully.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")


def delete_index(client: Elasticsearch) -> None:
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
        print(f"Index '{INDEX_NAME}' deleted successfully.")
    else:
        print(f"Index '{INDEX_NAME}' does not exist.")
