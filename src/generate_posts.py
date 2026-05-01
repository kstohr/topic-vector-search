"""
====================
GENERATE POSTS
====================

Generate synthetic posts on seven (7) example topics. This calls an LLM (in this
case, Ollama) to generate engaging social media posts that include relevant
emojis and keywords. The generated posts are saved to sample_posts.json and
serve as the dataset for the topic modeling and search pipeline.
"""

import json
import logging
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from openai import OpenAI

from src.config import OLLAMA_MODEL, OLLAMA_URL, REPO
from src.data_models import Post

ASSETS_DIR = REPO / "assets"

logger = logging.getLogger(__name__)

POSTS_PER_TOPIC = 20
RANDOM_POST_COUNT = 30

TOPICS = {
    "Cats, cats, cats": [
        "😺 Meow",
        "😻 Purr",
        "🐱 Kitty",
        "🛌 Catnap",
        "🐾 Feline",
        "🐈 Whiskers",
        "📅 Caturday",
        "👩‍👧‍👦 Catmom",
        "👨‍👧‍👦 Catdad",
        "🧶 Fluffy",
        "✨ Clawsome",
    ],
    "Music recommendations": [
        "🎧 Listen",
        "🎵 Music",
        "🔥 Must-Listen",
        "🎶 Vibes",
        "❤️ Favorite",
        "👌 Recommendation",
        "🙌 Check it out",
        "🆕 New Release",
        "⭐ Top Pick",
        "playlist",
    ],
    "Space Travel and Astronauts": [
        "🚀 Rocket",
        "👩‍🚀 Astronaut",
        "🌌 Cosmos",
        "🛸 SpaceX",
        "🌙 Artemis",
        "🌍 ISS",
        "NASA",
        "🔭 JWST",
        "Mars",
        "Orbit",
        "Launch",
        "Zero gravity",
    ],
    "Interior Design and Home Renovation": [
        "🛋️ InteriorDesign",
        "🏠 HomeReno",
        "🪞 Decor",
        "🌿 Biophilic",
        "🧱 Terracotta",
        "🛏️ Japandi",
        "Renovation",
        "Open shelving",
        "Color palette",
        "Texture layering",
        "DIY",
        "Minimalism",
    ],
    "San Francisco Fog": [
        "🌁 Fog",
        "🌫️ KarlTheFog",
        "🌉 GoldenGate",
        "🌧️ Misty",
        "❄️ Chilly",
        "🌥️ Overcast",
        "☁️ Cloudy",
        "🧥 BundleUp",
        "🌃 FoggyNight",
        "🏙️ SFWeather",
        "WeatherForecast",
    ],
    "California High Speed Rail": [
        "🚄 High Speed Rail",
        "HighSpeedRail",
        "train",
        "🛤️ Infrastructure",
        "🌉 SFtoLA",
        "Fresno",
        "🚧 Construction",
        "Funding",
        "Delays",
        "🚆 Bullet Train",
        "🌎 EcoFriendly",
        "📅 Timeline",
        "📈 Progress",
        "Federal Funding",
        "public transportation",
    ],
    "Open Water Swimming": [
        "🏊‍♂️ OpenWater",
        "🌊 Swim",
        "🏅 Endurance",
        "🚩 Buoy",
        "⏱️ Timing",
        "🥶 Cold Water",
        "🌅 SunriseSwim",
        "🏞️ Nature",
        "🐬 Wildlife",
        "💪 Challenge",
        "Alcatraz",
        "tide charts",
        "Triathlon",
    ],
}


def _random_datetime() -> datetime:
    """Return a random datetime within the past year."""
    start = datetime.now() - timedelta(days=365)
    return start + (datetime.now() - start) * random.random()


def _make_image_post(image_path: Path) -> Post:
    """Build an image-only Post. post_text is empty; caption is populated by preprocess.py."""
    return Post(
        post_id=str(uuid.uuid4()),
        post_author=f"user_{random.randint(1, 100)}",
        created_at=_random_datetime().isoformat(),
        modified_at=_random_datetime().isoformat(),
        post_text="",
        image_url=str(image_path.relative_to(REPO)),
        generated_topic=image_path.stem,
    )


def _make_post(post_text: str, generated_topic: str) -> Post:
    """Build a text Post with a random author and timestamps."""
    return Post(
        post_id=str(uuid.uuid4()),
        post_author=f"user_{random.randint(1, 100)}",
        created_at=_random_datetime().isoformat(),
        modified_at=_random_datetime().isoformat(),
        post_text=post_text,
        generated_topic=generated_topic,
    )


class PostGenerator:
    """Generates synthetic social media posts via an LLM."""

    def __init__(self) -> None:
        """Initialise the OpenAI-compatible client pointed at Ollama."""
        self.client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")

    def generate_topic_posts(self) -> list[Post]:
        """Generate posts for each defined topic in TOPICS."""
        posts = []
        for topic, keywords in TOPICS.items():
            logger.info(f"Generating posts for topic: {topic}")
            texts = self._call_llm_for_topic(topic, keywords)
            posts.extend(_make_post(text, topic) for text in texts)
        return posts

    def generate_image_posts(self) -> list[Post]:
        """Create one image-only post per image file found in assets/."""
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        images = [
            image_path
            for image_path in ASSETS_DIR.iterdir()
            if image_path.suffix.lower() in image_extensions
        ]
        logger.info(f"Creating {len(images)} image posts from {ASSETS_DIR.name}/.")
        return [_make_image_post(img) for img in sorted(images)]

    def generate_random_posts(self) -> list[Post]:
        """Generate random off-topic posts across distinct subjects."""
        logger.info(f"Generating {RANDOM_POST_COUNT} random posts.")
        texts = self._call_llm_for_random()
        return [_make_post(text, "noise") for text in texts]

    def _call_llm_for_topic(self, topic: str, keywords: list[str]) -> list[str]:
        """Call the LLM to generate posts for a single topic. Returns a list of post strings."""
        prompt = f"""
        Generate a list of {POSTS_PER_TOPIC} unique social media posts about {topic}.
        Include relevant emojis and keywords such as {",".join(keywords)}.
        Make each post engaging and relevant to the topic. Do not repeat the content of posts.
        Posts content should be unique. Vary the length of posts from one sentence to 10 sentences.
        Posts on more serious topics should be longer and do not need to include emojis.
        The tone of posts for more serious topics can range from excited to concerned and the
        sentiment can be positive, negative or neutral.
        Return each post as a separate line. Avoid numbering the posts or using any list formatting.
        """
        response = self.client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates creative social media posts.",  # noqa: E501
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1200,
            temperature=0.7,
        )
        if response.choices[0].message.content is None:
            raise ValueError(f"LLM returned no content for topic '{topic}'.")
        return [
            line.strip() for line in response.choices[0].message.content.split("\n") if line.strip()
        ]

    def _call_llm_for_random(self) -> list[str]:
        """Call the LLM to generate random off-topic posts. Returns a list of post strings."""
        prompt = f"""
        Generate a list of {RANDOM_POST_COUNT} unique social media posts on
        {RANDOM_POST_COUNT} distinct topics. Follow the guidelines below:
        - Include emojis to make them engaging.
        - Do not repeat the content of posts.
        - Vary the topic of the posts so that they are on random and distinct topics.
          (e.g., food, art, culture, travel, sports, technology, history, science, etc.)
        - Some posts can emulate short responses to other posts not included in the list.
        - Vary the length of posts from 1 sentence to 5 sentences.
        - Vary the tone of the posts from serious to lighthearted.
        - Vary the sentiment of the posts from positive, negative to neutral.
        Return each post as a separate line. Avoid numbering the posts or using any list formatting.
        """
        response = self.client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates social media posts.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1800,
            temperature=0.7,
        )
        if response.choices[0].message.content is None:
            raise ValueError("LLM returned no content for random posts.")
        return [
            line.strip() for line in response.choices[0].message.content.split("\n") if line.strip()
        ]

    def run(self) -> None:
        """Generate all posts and write them to sample_posts.json."""
        posts = (
            self.generate_topic_posts() + self.generate_random_posts() + self.generate_image_posts()
        )
        output_path = REPO / "sample_posts.json"
        with open(output_path, "w") as f:
            json.dump([post.model_dump(mode="json") for post in posts], f, indent=4)
        logger.info(f"Saved {len(posts)} posts to {output_path.name}.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    PostGenerator().run()
