import os
from pathlib import Path

# Paths
REPO = Path(__file__).parent.parent
OUTPUT = REPO / "output"

# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Vision model (image captioning)
VISION_MODEL_NAME = "Salesforce/blip-image-captioning-base"

# LLM for topic labeling
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OPENAI_MODEL = "gpt-4o-mini"
