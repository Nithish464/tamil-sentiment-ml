"""
Data scraping script for Tamil sentiment dataset.
Scrapes Tamil/Tanglish comments from YouTube and Twitter.

Prerequisites:
  - Twitter API key (developer.twitter.com)
  - YouTube Data API key (console.cloud.google.com)
  - pip install tweepy youtube-comment-downloader google-api-python-client
"""

import csv
import time
import pandas as pd
from pathlib import Path
from loguru import logger

DATA_DIR = Path(__file__).parent


def scrape_youtube_comments(video_ids: list, max_comments: int = 500) -> list:
    """
    Scrape comments from Tamil YouTube videos.
    Uses youtube-comment-downloader (no API key needed!).
    """
    try:
        from youtube_comment_downloader import YoutubeCommentDownloader
    except ImportError:
        logger.error("Install: pip install youtube-comment-downloader")
        return []

    downloader = YoutubeCommentDownloader()
    all_comments = []

    # Tamil movie trailers, cooking channels, product reviews
    default_video_ids = [
        "kJQP7kiw5Fk",   # Example Tamil video
        "9bZkp7q19f0",   # Example Tamil video
    ]
    video_ids = video_ids or default_video_ids

    for vid_id in video_ids:
        logger.info(f"Scraping YouTube video: {vid_id}")
        try:
            comments = downloader.get_comments_from_url(
                f"https://www.youtube.com/watch?v={vid_id}",
                sort_by=0
            )
            count = 0
            for comment in comments:
                if count >= max_comments:
                    break
                text = comment.get("text", "").strip()
                if text and len(text) > 5:
                    all_comments.append({"text": text, "source": "youtube"})
                    count += 1
            logger.info(f"  Collected {count} comments")
        except Exception as e:
            logger.warning(f"  Failed for {vid_id}: {e}")
        time.sleep(1)

    return all_comments


def scrape_twitter(query: str, max_tweets: int = 500, bearer_token: str = None) -> list:
    """
    Scrape Tamil tweets using Twitter API v2.
    Requires Twitter Developer Account bearer token.
    """
    if not bearer_token:
        logger.warning("Twitter bearer token not provided. Skipping Twitter scraping.")
        return []

    try:
        import tweepy
    except ImportError:
        logger.error("Install: pip install tweepy")
        return []

    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
    tweets_data = []

    # Tamil product review queries
    queries = [
        f"{query} lang:ta -is:retweet",
        "product review tamil -is:retweet",
        "நல்லா இருக்கு -is:retweet",
        "மோசமான -is:retweet",
    ]

    for q in queries:
        try:
            tweets = tweepy.Paginator(
                client.search_recent_tweets,
                query=q,
                max_results=100,
                tweet_fields=["text", "lang"]
            ).flatten(limit=max_tweets // len(queries))

            for tweet in tweets:
                if tweet.text:
                    tweets_data.append({"text": tweet.text, "source": "twitter"})

            logger.info(f"Query '{q}': {len(tweets_data)} tweets so far")
            time.sleep(2)
        except Exception as e:
            logger.warning(f"Twitter query failed: {e}")

    return tweets_data


def label_with_keywords(text: str) -> str:
    """
    Simple keyword-based labeling for initial dataset creation.
    Manual review recommended afterward.
    """
    text_lower = text.lower()

    positive_keywords = [
        "நல்ல", "அருமை", "super", "superb", "excellent", "amazing", "best",
        "love", "great", "wonderful", "romba nalla", "semma", "👍", "❤️", "😍",
        "worth", "happy", "satisfied", "perfect", "நன்றி", "thanks"
    ]
    negative_keywords = [
        "மோசம", "வேண்டாம", "waste", "bad", "worst", "terrible", "horrible",
        "useless", "fraud", "scam", "disappointed", "pathetic", "👎", "😡",
        "return", "refund", "broken", "damaged", "மோசமான"
    ]

    pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
    neg_count = sum(1 for kw in negative_keywords if kw in text_lower)

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"


def save_dataset(data: list, filename: str = "scraped_data.csv"):
    """Save scraped and labeled data to CSV."""
    output_path = DATA_DIR / filename
    labeled = []
    for item in data:
        item["label"] = label_with_keywords(item["text"])
        item["language"] = "tanglish"  # To be manually corrected
        labeled.append(item)

    df = pd.DataFrame(labeled)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"✅ Saved {len(df)} samples to {output_path}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    return df


def main():
    logger.info("Starting data collection...")

    # YouTube scraping (no API key needed)
    youtube_data = scrape_youtube_comments(
        video_ids=[],  # Add Tamil video IDs here
        max_comments=300
    )

    # Twitter scraping (needs API key)
    # twitter_data = scrape_twitter(
    #     query="product review",
    #     bearer_token="YOUR_BEARER_TOKEN"
    # )

    all_data = youtube_data  # + twitter_data

    if all_data:
        df = save_dataset(all_data, "scraped_data.csv")
        logger.info(f"\n✅ Collected {len(df)} samples total!")
    else:
        logger.info("Using sample_data.csv for training (add video IDs to collect more)")


if __name__ == "__main__":
    main()
