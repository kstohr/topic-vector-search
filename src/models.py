import re
from datetime import datetime, timezone
from typing import List, Optional

import emoji
import numpy as np
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer


# Define the Pydantic model for posts
class PostDocument(BaseModel):
    post_id: str
    post_author: str
    created_at: datetime
    modified_at: datetime
    post_text: str
    doc_embedding: Optional[List[float]] = Field(default_factory=list)

    @field_validator("created_at", "modified_at", mode="before")
    @classmethod
    def set_datetime_to_utc(cls, value):
        # Parse the datetime string, convert to UTC, and return in ISO format
        dt = datetime.fromisoformat(value)
        if (
            dt.tzinfo is None
        ):  # If no timezone info, assume it's local and convert to UTC
            dt = dt.replace(tzinfo=timezone.utc)
        else:  # Convert to UTC if timezone info is present
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat()

    def preprocess_text(self):
        """
        Passed to model pipeline. Standard pre-processing of text after cleaning,
        prior to modeling. Does not include sentence splitting. If sentence
        embeddings are needed use `preprocess_sentences`.
        """
        # Replace emojis with their textual descriptions
        text = emoji.demojize(self.post_text)  # noqa
        # Lowercase the text
        text = text.lower()
        # Remove punctuation but preserve contractions and compound words
        text = re.sub(r"[^\w\s'-]", "", text)
        # Remove extra whitespaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # async def preprocess_sentences(self) -> list[str]:
    #     """
    #     Splits the text into sentences. Pre-processes each sentence. Returns a list
    #     of cleaned sentences.
    #     """

    #     # Use spaCy to segment text into sentences
    #     nlp = spacy.load("en_core_web_sm")
    #     doc: list[Span] = nlp(self.post_text)
    #     split_text = [sent.text for sent in doc.sents]
    #     sentences = []
    #     for sent in split_text:
    #         sentences.append(preprocess_text(sent))
    #     return sentences

    def preprocess_sentences(self) -> list[str]:
        """
        Splits the text into sentences using regex (Normally SpaCy).
        Pre-processes the input text
        by lowercasing, removing numbers, extra whitespaces, and replacing emojis
        with their textual descriptions.
        """

        # Split text into sentences using regex (matches periods, exclamations, and questions)
        split_text = re.split(r"(?<=[.!?])\s+", self.post_text)

        # Apply preprocessing to each sentence
        sentences = [self.preprocess_text(sent) for sent in split_text if sent.strip()]
        return sentences

    # Create embeddings using SentenceTransformers and store them in the txt_embedding field
    def create_embeddings(self):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(self.preprocess_sentences())
        self.doc_embedding = np.mean(embeddings, axis=0).tolist()

    class Config:
        arbitrary_types_allowed = True
