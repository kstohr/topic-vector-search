import json
import logging
import os
from typing import List, Tuple

import hdbscan
import numpy as np
import openai
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.representation import OpenAI as BertTopicOpenAI
from bertopic.vectorizers import ClassTfidfTransformer
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from .models import PostDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RANDOM_SEED = 99

# Load environment variables for OpenAI API key, organization, and project
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.project = os.getenv("OPENAI_PROJECT")


class TopicModeler:
    def __init__(self, index_name: str, output_path: str = "output"):
        self.index_name = index_name
        # The embedding model must match the model used to store the embeddings on Elastic.
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.opensearch_client = OpenSearch(
            hosts=[{"host": "localhost", "port": 9200}], http_compress=True
        )
        self.topic_model = None
        self.random_state = RANDOM_SEED
        self.doc_index: dict[str, PostDocument] = {}
        self.output_path = output_path
        self.ensure_output_path_exists()

    def ensure_output_path_exists(self):
        """Create the output directory if it does not exist."""
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            logger.info(f"Created output directory at: {self.output_path}")
        else:
            logger.info(f"Output directory already exists: {self.output_path}")

    def retrieve_post_documents(self) -> dict[str, PostDocument]:
        """Retrieve all post documents from OpenSearch. Returns an index of post_id to PostDocument."""
        logger.info("Retrieving embeddings from OpenSearch.")
        query = {"size": 1000, "query": {"match_all": {}}}
        response = self.opensearch_client.search(index=self.index_name, body=query)
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            post = PostDocument(**source)
            self.doc_index[post.post_id] = post
        logger.info(f"Retrieved {len(self.doc_index)} post documents.")
        return self.doc_index

    def extract_texts(self) -> List[str]:
        """Extract the preprocessed text from the PostDocuments."""
        logger.info("Preprocessing text.")
        texts = [doc.preprocess_text() for doc in self.doc_index.values()]
        return texts

    def extract_embeddings(self) -> np.ndarray:
        """Extract embeddings from the post documents."""
        logger.info("Extracting embeddings.")
        embeddings = [doc.doc_embedding for doc in self.doc_index.values()]
        return np.array(embeddings)

    def train_topic_model(
        self, texts: List[str], embeddings: np.ndarray
    ) -> Tuple[List[int], np.ndarray | None]:
        """Train BERTopic model using the embeddings and texts."""
        logger.info("Training BERTopic model.")

        embedding_model = self.embedding_model
        vectorizer_model = CountVectorizer(
            stop_words="english",
            # If needed, set cut-off thresholds for document frequency
            # this can help to remove very common or very rare words in the
            # topic representation
            # max_df=0.95,
            # min_df=0.01
        )
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=12,  # Ensures clusters need at least 10 points to form a distinct group
            min_samples=5,  # Minimum number of samples in a cluster
            metric="euclidean",  # Does not support cosine distance with the standard backend, so we use Euclidean
            cluster_selection_method="eom",
            cluster_selection_epsilon=0.001,  # 0.01 # Make cluster selection more conservative to ensure clusters are more cohesive.
            prediction_data=True,
        )

        keybert_model = KeyBERTInspired(
            top_n_words=10,
            nr_repr_docs=5,
            nr_samples=500,
            nr_candidate_words=100,
            random_state=RANDOM_SEED,
        )

        openai_model = BertTopicOpenAI(
            client=openai,
            model="gpt-3.5-turbo",
            exponential_backoff=True,
            chat=True,
            prompt="""
                I have a topic that contains the following documents:
                [DOCUMENTS]
                The topic is described by the following keywords: [KEYWORDS]

                Based on the information above, extract a short but highly descriptive topic
                label of at most 3 words. Make sure it is in the following format:
                topic: <topic label>
                """,
        )

        # Train BERTopic with HDBSCAN and SentenceTransformer embeddings
        self.topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ClassTfidfTransformer(),  # bm25_weighting=True
            representation_model={
                "KeyBERT": keybert_model,
                "OpenAI": openai_model,
            },
            calculate_probabilities=True,
        )

        self.topics, self.probabilities = self.topic_model.fit_transform(
            texts, embeddings
        )

        logger.info("Finished training BERTopic model with OpenAI representations.")
        return self.topics, self.probabilities

    def generate_visualizations(self):
        """Generate visualizations using BERTopic."""
        if not self.topic_model:
            logger.error("Topic model has not been trained yet.")
            return
        logger.info("Generating visualizations.")
        # Save visualizations to the output path
        self.topic_model.visualize_topics().write_html(
            os.path.join(self.output_path, "topic_visualization.html")
        )
        self.topic_model.visualize_barchart().write_html(
            os.path.join(self.output_path, "barchart.html")
        )
        self.topic_model.visualize_heatmap().write_html(
            os.path.join(self.output_path, "heatmap.html")
        )
        self.topic_model.visualize_hierarchy().write_html(
            os.path.join(self.output_path, "hierarchy.html")
        )
        self.topic_model.visualize_term_rank().write_html(
            os.path.join(self.output_path, "term_rank.html")
        )
        logger.info("Visualizations generated and saved.")

    def store_model_data(
        self, topics: List[int], probabilities: np.ndarray, texts: List[str]
    ):
        """Store the model and associated data."""
        logger.info("Storing model data.")
        if not self.topic_model:
            logger.error("Topic model has not been trained yet.")
            return
        # Save model
        self.model_path = os.path.join(self.output_path, "bertopic_model")
        # Create the directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)

        self.topic_model.save(
            path=self.model_path,
            serialization="pytorch",
            save_ctfidf=True,
            save_embedding_model=self.embedding_model,
        )
        logger.info(f"Model saved to {self.model_path}")

        # Save topics and probabilities
        with open(os.path.join(self.output_path, "topics.json"), "w") as f:
            json.dump(topics, f)

        # Convert the numpy array to a list
        probabilities_list = probabilities.tolist()

        # Save the probabilities as JSON
        with open(os.path.join(self.output_path, "probabilities.json"), "w") as f:
            json.dump(probabilities_list, f)
        logger.info("Topics and probabilities saved.")

        # Save texts with their associated topics
        df = pd.DataFrame(
            {"post_id": self.doc_index.keys(), "text": texts, "topic_id": topics}
        )
        csv_path = os.path.join(self.output_path, "topic_assignments.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Text with topics saved to {csv_path}")

        # Save the doc index as JSON
        doc_index_path = os.path.join(self.output_path, "doc_index.json")
        with open(doc_index_path, "w") as f:
            json.dump(
                {k: v.model_dump() for k, v in self.doc_index.items()}, f, default=str
            )

    def load_topic_model(self) -> BERTopic:
        """Load the BERTopic model."""
        logger.info("Loading BERTopic model.")
        model_path = os.path.join(self.output_path, "bertopic_model")
        self.topic_model = BERTopic.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return self.topic_model

    def run(self):
        """Run all steps to train and save the topic model."""
        # Step 1: Retrieve embeddings from OpenSearch
        self.retrieve_post_documents()

        # Step 2: Preprocess the texts
        preprocessed_texts = self.extract_texts()

        # Step 3: Generate embeddings using SentenceTransformers
        embeddings = self.extract_embeddings()

        # Step 4: Train the BERTopic model with OpenAI as the representation model
        topics, probabilities = self.train_topic_model(preprocessed_texts, embeddings)

        # Step 5: Generate visualizations
        self.generate_visualizations()

        # Step 6: Store the model and data
        self.store_model_data(topics, probabilities, preprocessed_texts)


# Example usage
if __name__ == "__main__":
    topic_modeler = TopicModeler(index_name="post_docs")
    topic_modeler.run()

    # Step 7: Load the model (for demonstration or further use)
    loaded_model = topic_modeler.load_topic_model()
    logger.info("Model loaded successfully.")
