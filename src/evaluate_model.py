"""
Evaluation script for Tamil sentiment model.
Generates classification report, confusion matrix, and per-language breakdown.
"""

import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from preprocess_data import load_and_prepare
from dataset import TamilSentimentDataset
from torch.utils.data import DataLoader

MODELS_DIR  = Path(__file__).parent.parent / "models"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["negative", "neutral", "positive"]


def load_model():
    model_path = MODELS_DIR / "best_model"
    with open(MODELS_DIR / "metadata.json") as f:
        metadata = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(DEVICE)
    model.eval()
    return model, tokenizer, metadata


def predict_batch(model, tokenizer, texts: list, max_length: int = 128) -> list:
    """Get predictions for a list of texts."""
    dataset = TamilSentimentDataset(texts, [0] * len(texts), tokenizer, max_length)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
            )
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    return all_preds, all_probs


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix — Tamil Sentiment Model", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✅ Confusion matrix saved!")


def per_language_analysis(df_test: pd.DataFrame, preds: list):
    """Breakdown F1 score by language (Tamil / English / Tanglish)."""
    df_test = df_test.copy()
    df_test["pred"] = preds

    logger.info("\n📊 Per-Language F1 Score:")
    results = {}
    for lang in df_test["language"].unique():
        subset = df_test[df_test["language"] == lang]
        if len(subset) < 5:
            continue
        f1 = f1_score(subset["label_id"], subset["pred"], average="macro")
        results[lang] = round(f1, 4)
        logger.info(f"  {lang:<12}: F1 = {f1:.4f} ({len(subset)} samples)")

    return results


def main():
    logger.info("Loading model and test data...")
    model, tokenizer, metadata = load_model()
    data = load_and_prepare()

    test_df = data["test"]
    preds, probs = predict_batch(model, tokenizer, test_df["text"].tolist())
    true_labels  = test_df["label_id"].tolist()

    logger.info("\n" + "="*50)
    logger.info("CLASSIFICATION REPORT")
    logger.info("="*50)
    logger.info("\n" + classification_report(true_labels, preds, target_names=LABEL_NAMES))

    plot_confusion_matrix(true_labels, preds)
    lang_results = per_language_analysis(test_df, preds)

    with open(REPORTS_DIR / "eval_results.json", "w") as f:
        json.dump({
            "overall_f1": round(f1_score(true_labels, preds, average="macro"), 4),
            "per_language_f1": lang_results,
        }, f, indent=2)

    logger.info("\n✅ Evaluation complete! Check reports/ folder.")


if __name__ == "__main__":
    main()
