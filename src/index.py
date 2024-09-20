from opensearchpy import OpenSearch

# Define the OpenSearch index
INDEX_NAME = "post_docs"
INDEX_BODY = {
    "settings": {"index.knn": True, "index": {"number_of_shards": 1}},
    "mappings": {
        "properties": {
            "post_id": {
                "type": "keyword",
                "doc_values": True,
                "index": True,
                "norms": False,
            },
            "post_author": {"type": "keyword"},
            "created_at": {"type": "date", "format": "strict_date_time"},
            "modified_at": {"type": "date", "format": "strict_date_time"},
            "post_text": {"type": "text"},
            "doc_embedding": {
                "type": "knn_vector",
                "dimension": 384,  # Match the dimensions used in SentenceTransformers
            },
        }
    },
}


# Function to create the OpenSearch index
def create_index(opensearch_client):
    if not opensearch_client.indices.exists(INDEX_NAME):
        opensearch_client.indices.create(index=INDEX_NAME, body=INDEX_BODY)
        print(f"Index '{INDEX_NAME}' created successfully.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")


# Function to teardown the OpenSearch index 
def delete_index(opensearch_client): 
    if opensearch_client.indices.exists(INDEX_NAME):
        opensearch_client.indices.delete(index=INDEX_NAME)
        print(f"Index '{INDEX_NAME}' deleted successfully.")
    else:
        print(f"Index '{INDEX_NAME}' does not exists.")
            
