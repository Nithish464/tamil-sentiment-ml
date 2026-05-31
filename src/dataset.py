"""
PyTorch Dataset class for Tamil sentiment classification.
Handles tokenization for IndicBERT/mBERT models.
"""

import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger


class TamilSentimentDataset(Dataset):
    """
    Dataset for Tamil/English/Tanglish sentiment classification.
    Compatible with IndicBERT and mBERT tokenizers.
    """

    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


def create_dataloaders(data: dict, tokenizer, batch_size: int = 16, max_length: int = 128):
    """Create PyTorch DataLoaders for train/val/test splits."""
    from torch.utils.data import DataLoader

    loaders = {}
    for split in ["train", "val", "test"]:
        df = data[split]
        dataset = TamilSentimentDataset(
            texts=df["text"].tolist(),
            labels=df["label_id"].tolist(),
            tokenizer=tokenizer,
            max_length=max_length,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        logger.info(f"{split} DataLoader: {len(dataset)} samples, {len(loaders[split])} batches")

    return loaders
