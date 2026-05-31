"""
Data preprocessing pipeline for Tamil sentiment analysis.
Handles Tamil script, Tanglish (romanized Tamil), and English.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from loguru import logger

DATA_DIR = Path(__file__).parent


TANGLISH_NORMALIZATIONS = {
    "romba": "மிகவும்",
    "nalla": "நல்ல",
    "illa": "இல்ல",
    "iruку": "இருக்கு",
    "paaru": "பாரு",
    "sollu": "சொல்லு",
    "semma": "சூப்பர்",
    "da": "",
    "bro": "",
    "machi": "",
}


def clean_text(text: str) -> str:
    """Clean and normalize text while preserving Tamil characters."""
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)
    # Remove excessive punctuation but keep Tamil punct
    text = re.sub(r"[!]{2,}", "!", text)
    text = re.sub(r"[.]{2,}", ".", text)
    # Remove extra whitespace
    text = " ".join(text.split())

    return text.strip()


def detect_language(text: str) -> str:
    """Detect if text is Tamil, English, or Tanglish."""
    tamil_chars = len(re.findall(r"[\u0B80-\u0BFF]", text))
    total_chars = len(text.replace(" ", ""))

    if total_chars == 0:
        return "unknown"

    tamil_ratio = tamil_chars / total_chars

    if tamil_ratio > 0.4:
        return "tamil"
    elif tamil_ratio > 0.05:
        return "tanglish"
    else:
        return "english"


def load_and_prepare(
    data_path: str = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> dict:
    """
    Load dataset and prepare train/val/test splits.
    Falls back to sample_data.csv if no path provided.
    """
    if data_path is None:
        data_path = DATA_DIR / "sample_data.csv"
        logger.info(f"Using sample dataset: {data_path}")

    df = pd.read_csv(data_path, encoding="utf-8")
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")

    # Clean text
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 3].reset_index(drop=True)

    # Auto-detect language if not present
    if "language" not in df.columns:
        df["language"] = df["text"].apply(detect_language)

    # Encode labels
    label_map = {"positive": 2, "neutral": 1, "negative": 0}
    df["label_id"] = df["label"].map(label_map)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)

    # Split
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label_id"], random_state=random_state
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, stratify=train_df["label_id"], random_state=random_state
    )

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    return {
        "train": train_df.reset_index(drop=True),
        "val":   val_df.reset_index(drop=True),
        "test":  test_df.reset_index(drop=True),
        "label_map": label_map,
        "id_to_label": {v: k for k, v in label_map.items()},
    }


if __name__ == "__main__":
    data = load_and_prepare()
    print(f"\n✅ Data ready!")
    print(f"   Train: {len(data['train'])}")
    print(f"   Val  : {len(data['val'])}")
    print(f"   Test : {len(data['test'])}")
