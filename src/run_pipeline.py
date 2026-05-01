"""
End-to-end pipeline: preprocessing → topic modeling.
"""

from src.preprocess import PreprocessingPipeline
from src.topic_model import TopicModeler
from src.es_index import delete_index, get_es_client


def pipeline():
    """Run preprocessing then topic modeling in sequence."""

    print("Deleting existing Elasticsearch index (if any)...")
    es_client = get_es_client()
    delete_index(client=es_client)

    print("Running preprocessing pipeline...")
    PreprocessingPipeline().run()
    print("Preprocessing complete.\n")

    print("Running topic modeling pipeline...")
    TopicModeler().run()
    print("Topic modeling complete.\n")

    print("Pipeline complete. Run: `uv run streamlit run app.py`to start the demo app.")


if __name__ == "__main__":
    pipeline()
