"""
Generate posts on six (6) example topics.
"""

import json
import logging
import os
import random
import uuid
from datetime import datetime, timedelta

import openai
from openai.types.chat.chat_completion import ChatCompletion

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_MODEL = "gpt-3.5-turbo"
RANDOM_SEED = 99

# Load environment variables for OpenAI API key, organization, and project
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.project = os.getenv("OPENAI_PROJECT")

# Define the topics and keywords
topics = {
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
    "Social Activism": [
        "✊🏽 Solidarity",
        "📢 SpeakUp",
        "🌍 Change",
        "🗳️ Vote",
        "⚖️ Justice",
        "📣 Activism",
        "🤝 Community",
        "🚫 NoHate",
        "✊🏿 BlackLivesMatter",
        "🌈 TransGender Rights",
        "Climate Change",
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

number_of_posts = 20  # Number of posts per topic


# Function to generate a random datetime for the post creation and modification within the past year
def random_datetime():
    start = datetime.now() - timedelta(days=365)
    end = datetime.now()
    return start + (end - start) * random.random()


# Function to call OpenAI API to generate posts
def generate_posts_list(topic, keywords):
    prompt = f"""
    Generate a list of {number_of_posts} unique social media posts about {topic}.
    Include relevant emojis and keywords such as {','.join(keywords)}. Make each
    post engaging and relevant to the topic. Do not repeat the content of posts.
    Posts content should be unique. Vary the length of
    posts from one sentence to 10 sentences. Posts on more serious topics should
    be longer and do not need to include emojis. The tone of posts for more
    serious topics can range from excited to concerned and the sentiment can be
    positive, negative or neutral. Return each post as a separate line. Avoid
    numbering the posts or using any list formatting.
    """

    response: ChatCompletion = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates creative social media posts.",
            },
            {"role": "user", "content": prompt},
        ],
        seed=RANDOM_SEED,
        max_tokens=1200,
        temperature=0.7,
    )

    # Split the response into individual posts
    if response.choices[0].message.content is None:
        logger.error("Failed to generate random posts.")
        raise ValueError("Failed to generate random posts.")
    post_texts = response.choices[0].message.content.split("\n")
    return [post.strip() for post in post_texts if post.strip()]


def main():
    # Generate posts
    posts = []

    # Generate posts for each topic
    for topic, keywords in topics.items():
        logger.info(f"Generating posts for topic: {topic}")
        post_texts = generate_posts_list(topic, keywords)
        for post_text in post_texts:
            post = {
                "post_id": str(uuid.uuid4()),
                "post_author": f"user_{random.randint(1, 100)}",
                "created_at": random_datetime().isoformat(),
                "modified_at": random_datetime().isoformat(),
                "post_text": post_text,
                "txt_embedding": [],  # Embedding field left blank
            }
            posts.append(post)

    # Generate random posts for other topics
    number_of_random_posts = 30  # Number of posts with random topics
    random_topic_prompt = f"""
    Generate a list of {number_of_random_posts} unique social media posts on
    {number_of_random_posts} distinct topics. Follow the guidelines below in
    creating a set of posts:
    - Include emojis to make them
    engaging.
    - Do not repeat the content of posts.
    - Vary the topic of the posts so that they are on random and distinct topics.
    (e.g., food, art, culture, travel, sports, self-help, technology, history, science, geology, etc.)
    - Some posts can emulate short responses to other posts not included in the list.
    - Vary the length of posts from 1 sentence to 5 sentences.
    - Vary the tone of the posts from serious to lighthearted
    - Vary the sentiment of the posts from positive, negative to neutral.

    Return each post as a separate line. Avoid numbering the posts or using any list formatting.
    """
    logger.info("Generating random posts")
    response: ChatCompletion = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates social media posts.",
            },
            {"role": "user", "content": random_topic_prompt},
        ],
        seed=RANDOM_SEED,
        max_tokens=1800,  # Adjusted to accommodate multiple posts
        temperature=0.7,
    )

    # Split the random topic posts into individual entries
    if response.choices[0].message.content is None:
        logger.error("Failed to generate random posts.")
        raise ValueError("Failed to generate random posts.")
    random_posts = response.choices[0].message.content.split("\n")
    random_posts = [post.strip() for post in random_posts if post.strip()]

    for post_text in random_posts:
        post = {
            "post_id": str(uuid.uuid4()),
            "post_author": f"user_{random.randint(1, 100)}",
            "created_at": random_datetime().isoformat(),
            "modified_at": random_datetime().isoformat(),
            "post_text": post_text,
            "txt_embedding": [],  # Embedding field left blank
        }
        posts.append(post)

    # Save the posts to a JSON file
    output_path = "sample_posts.json"
    with open(output_path, "w") as file:
        json.dump(posts, file, indent=4)

    print(f"Sample posts generated and saved to {output_path}")


if __name__ == "__main__":
    main()
