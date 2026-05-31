"""
FastAPI application — Tamil Sentiment Analysis API
Run: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""

import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .schemas import (
    SentimentRequest, SentimentResponse,
    BatchRequest, HealthResponse, MetricsResponse
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import load_artifacts, predict_single

# Global model state
_model = _tokenizer = _metadata = None


def detect_language(text: str) -> str:
    tamil_chars = len(re.findall(r"[\u0B80-\u0BFF]", text))
    ratio = tamil_chars / max(len(text.replace(" ", "")), 1)
    if ratio > 0.4:   return "tamil"
    if ratio > 0.05:  return "tanglish"
    return "english"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _metadata
    logger.info("Loading model...")
    _model, _tokenizer, _metadata = load_artifacts()
    logger.info("✅ Model ready!")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Tamil & English Sentiment Analysis API",
    description="""
## 🌐 Multilingual Sentiment Analyzer

Detect sentiment in **Tamil, English, and Tanglish** (code-mixed).

### Supported Languages
- **Tamil** (தமிழ்) — native script
- **English** — standard
- **Tanglish** — Tamil written in Roman script (romba nalla, semma, etc.)

### Endpoints
- **POST /predict** — Single text sentiment
- **POST /predict/batch** — Batch analysis
- **GET /health** — Health check
- **GET /metrics** — Model performance
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        model_name=_metadata["model_name"] if _metadata else "not loaded",
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
async def metrics():
    if not _metadata:
        raise HTTPException(status_code=404, detail="Train model first!")
    return MetricsResponse(
        test_accuracy=_metadata.get("test_accuracy", 0),
        test_f1=_metadata.get("test_f1", 0),
        best_val_f1=_metadata.get("best_val_f1", 0),
        model_name=_metadata["model_name"],
    )


@app.post("/predict", response_model=SentimentResponse, tags=["Sentiment"])
async def predict(request: SentimentRequest):
    """
    Analyze sentiment of Tamil/English/Tanglish text.

    Returns: sentiment label, confidence score, and per-class probabilities.
    """
    if not _model:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")
    try:
        result = predict_single(request.text, _model, _tokenizer, _metadata)
        result["text"] = request.text
        result["detected_language"] = detect_language(request.text)
        return SentimentResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=list[SentimentResponse], tags=["Sentiment"])
async def predict_batch(request: BatchRequest):
    """Analyze sentiment for a batch of texts (max 100)."""
    if not _model:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Max batch size is 100")
    try:
        results = []
        for text in request.texts:
            result = predict_single(text, _model, _tokenizer, _metadata)
            result["text"] = text
            result["detected_language"] = detect_language(text)
            results.append(SentimentResponse(**result))
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
