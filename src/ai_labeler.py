"""
Build an AI labeler for the BERTopic model to represent each topic with a short
descriptive label.  Use an LLM if available, otherwise fall back to KeyBERT
keywords.

"""

import logging
import os

from bertopic.representation import OpenAI as BertTopicOpenAI

from src.config import (
    OLLAMA_MODEL,
    OLLAMA_URL,
    OPENAI_MODEL,
)

logger = logging.getLogger(__name__)


def build_llm_representation(prompt: str) -> BertTopicOpenAI | None:
    """
    Return a BERTopic OpenAI representation model pointed at the best available backend:
      1. Ollama (local, no key needed)
      2. OpenAI (OPENAI_API_KEY env var)
      3. None → KeyBERT-only labels
    """
    import httpx
    from openai import OpenAI

    try:
        resp = httpx.get(OLLAMA_URL.replace("/v1", "/api/tags"), timeout=2)
        if resp.status_code == 200:
            logger.info(f"Ollama reachable at {OLLAMA_URL} — using {OLLAMA_MODEL}")
            client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
            return BertTopicOpenAI(
                client=client,
                model=OLLAMA_MODEL,
                exponential_backoff=True,
                chat=True,
                prompt=prompt,
                nr_docs=5,
            )
    except Exception:
        pass

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        logger.info("Using OpenAI API for topic labels.")
        client = OpenAI(
            api_key=api_key,
            organization=os.getenv("OPENAI_ORGANIZATION"),
            project=os.getenv("OPENAI_PROJECT"),
        )
        return BertTopicOpenAI(
            client=client,
            model=OPENAI_MODEL,
            exponential_backoff=True,
            chat=True,
            prompt=prompt,
            nr_docs=5,
        )

    logger.warning("No LLM available — topic labels will use KeyBERT keywords only.")
    return None
