import re
from datetime import UTC, datetime

import emoji
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME


# Define the Pydantic model for posts
class PostDocument(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    post_id: str
    post_author: str
    created_at: datetime
    modified_at: datetime
    post_text: str
    likes: int = 0
    image_url: str | None = None
    image_caption: str | None = None  # populated by LO5 vision exercise
    generated_topic: str | None = None
    doc_embedding: list[float] = Field(default_factory=list)

    @field_validator("created_at", "modified_at", mode="before")
    @classmethod
    def set_datetime_to_utc(cls, value: str) -> str:
        # Parse the datetime string, convert to UTC, and return in ISO format
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:  # noqa: SIM108
            dt = dt.replace(tzinfo=UTC)
        else:
            dt = dt.astimezone(UTC)
        return dt.isoformat()

    def preprocess_text(self, text: str | None = None) -> str:
        """
        Passed to model pipeline. Standard pre-processing of text after cleaning,
        prior to modeling. Does not include sentence splitting. If sentence
        embeddings are needed use `preprocess_sentences`.
        """
        if text is None:
            text = self.post_text
        text = emoji.demojize(text)  # noqa
        text = text.lower()
        text = re.sub(r"[^\w\s'-]", "", text)
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
    def create_embeddings(self) -> None:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embeddings = model.encode(self.preprocess_sentences())
        self.doc_embedding = np.mean(embeddings, axis=0).tolist()
