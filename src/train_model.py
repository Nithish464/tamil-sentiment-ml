"""
Fine-tune mBERT for Tamil sentiment classification.
Optimized for CPU training - fast and lightweight.
"""

import json
import torch
import numpy as np
from pathlib import Path
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, accuracy_score
from loguru import logger
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from preprocess_data import load_and_prepare
from dataset import create_dataloaders

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "bert-base-multilingual-cased"
NUM_LABELS = 3
MAX_LENGTH = 64    # Reduced from 128 — faster!
BATCH_SIZE = 8     # Reduced from 16 — less RAM!
EPOCHS     = 2     # Reduced from 5 — faster!
LR         = 2e-5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer():
    logger.info(f"Loading {MODEL_NAME}...")
    logger.info(f"Device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )
    model = model.to(DEVICE)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def train_epoch(model, loader, optimizer, scheduler) -> dict:
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    return {
        "loss":     total_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1":       f1_score(all_labels, all_preds, average="macro"),
    }


def eval_epoch(model, loader, split: str = "val") -> dict:
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split}", leave=False):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return {
        "loss":     total_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1":       f1_score(all_labels, all_preds, average="macro"),
        "preds":    all_preds,
        "labels":   all_labels,
    }


def train():
    logger.info("=" * 60)
    logger.info("TAMIL SENTIMENT — FINE-TUNING mBERT (CPU optimized)")
    logger.info("=" * 60)

    data = load_and_prepare()
    model, tokenizer = load_model_and_tokenizer()
    loaders = create_dataloaders(data, tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(loaders["train"]) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    best_val_f1 = 0

    for epoch in range(1, EPOCHS + 1):
        logger.info(f"\nEpoch {epoch}/{EPOCHS}")
        train_metrics = train_epoch(model, loaders["train"], optimizer, scheduler)
        val_metrics   = eval_epoch(model, loaders["val"], "val")

        logger.info(f"  Train → Loss: {train_metrics['loss']:.4f} | F1: {train_metrics['f1']:.4f}")
        logger.info(f"  Val   → Loss: {val_metrics['loss']:.4f}  | F1: {val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            model.save_pretrained(MODELS_DIR / "best_model")
            tokenizer.save_pretrained(MODELS_DIR / "best_model")
            logger.info(f"  ✅ Best model saved! Val F1: {best_val_f1:.4f}")

    logger.info("\nRunning final test evaluation...")
    test_model = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR / "best_model")
    test_model = test_model.to(DEVICE)
    test_metrics = eval_epoch(test_model, loaders["test"], "test")

    logger.info(f"\n{'='*50}")
    logger.info(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    logger.info(f"  F1 Macro : {test_metrics['f1']:.4f}")

    metadata = {
        "model_name":    MODEL_NAME,
        "num_labels":    NUM_LABELS,
        "id_to_label":   data["id_to_label"],
        "label_map":     data["label_map"],
        "test_accuracy": round(test_metrics["accuracy"], 4),
        "test_f1":       round(test_metrics["f1"], 4),
        "best_val_f1":   round(best_val_f1, 4),
        "epochs":        EPOCHS,
        "max_length":    MAX_LENGTH,
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\n✅ Training complete!")


if __name__ == "__main__":
    train()