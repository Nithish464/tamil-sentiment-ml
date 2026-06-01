"""Shared utility functions."""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODELS_DIR = Path(__file__).parent.parent / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_artifacts():
    with open(MODELS_DIR / "metadata.json") as f:
        metadata = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3)
    model = model.to(DEVICE)
    model.eval()
    return model, tokenizer, metadata


def predict_single(text: str, model, tokenizer, metadata: dict) -> dict:
    """Predict sentiment for a single text."""
    import time
    t0 = time.perf_counter()

    inputs = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        pred_id = int(probs.argmax())

    elapsed_ms = (time.perf_counter() - t0) * 1000
    id_to_label = {int(k): v for k, v in metadata["id_to_label"].items()}

    return {
        "sentiment":        id_to_label[pred_id],
        "confidence":       round(float(probs.max()), 4),
        "probabilities": {
            "negative": round(float(probs[0]), 4),
            "neutral":  round(float(probs[1]), 4),
            "positive": round(float(probs[2]), 4),
        },
        "inference_time_ms": round(elapsed_ms, 2),
    }
