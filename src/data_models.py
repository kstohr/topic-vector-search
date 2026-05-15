"""
====================
DATA MODELS
====================
Pydantic models for raw posts (Post) and processed post documents
(PostDocument).
"""

import re  # noqa: F401 - exercise placeholder
from datetime import UTC, datetime

import emoji  # noqa: F401 - exercise placeholder
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Post(BaseModel):
    """
    Raw unprocessed social media post as generated or ingested. This is what
    might be stored in a database for a post. The fields here are based on
    typical social media post attributes.
    """

    post_id: str
    post_author: str
    created_at: datetime
    modified_at: datetime
    post_text: str
    likes: int = 0
    image_url: str | None = None
    generated_topic: str | None = None  # Reference only

    @field_validator("created_at", "modified_at", mode="before")
    @classmethod
    def set_datetime_to_utc(cls, value: str) -> str:
        """Normalise datetime strings to UTC ISO format."""
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:  # noqa: SIM108
            dt = dt.replace(tzinfo=UTC)
        else:
            dt = dt.astimezone(UTC)
        return dt.isoformat()


# Inherits from Post, so all fields of Post are also fields of PostDocument
class PostDocument(Post):
    """
    Structured post document. Inherits all fields from Post, with additional fields
    for downstream modeling.
    - Deconstructs unstructured text and elements of the raw post into structured fields.
    - Methods to preprocess text
        - emoji conversion
        - handling urls, links
        - removing @mentions
        - etc. depending on the use case
    Note: These methods may vary based on the documents being processed and the
    downstream modeling task.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_caption: str | None = None
    doc_embedding: list[float] = Field(default_factory=list)

    def preprocess_text(self, text: str | None = None) -> str:
        """
        Handled during preprocessing in `preprocess.py`. Cleans attributes of
        the text that require transformation prior to tokenization. Does not
        include sentence splitting. If sentence embeddings are needed, see
        `preprocess_sentences` below.
        """
        ### EXERCISE ###
        # Add or remove preprocessing steps to clean text
        # At a minimum, remove URLs and convert emojis to text. You can add more steps as needed.
        # Run `uv run pytest tests/test_data_models.py -k PreprocessText`
        # When you've completed, the implementation, return to the notebook to
        # generate an embedding for the post.
        if text is None:
            text = self.post_text
        return text.strip()

    ##############################
    # Commented out for simplicity and due to SpaCy dependency
    # Try this out if you have time.
    # Run: `uv add spacy` and `python -m spacy download en_core_web_sm` to use
    # the sentence splitting method below.
    # Helpful for longer documents where you want to embed sentences instead of the whole text.
    ##############################
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
