"""Pydantic schemas for Tamil Sentiment API."""

from pydantic import BaseModel, Field
from typing import Optional


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, description="Tamil/English/Tanglish text")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "romba nalla product da! friends ku recommend pannuven"
            }
        }


class SentimentResponse(BaseModel):
    text: str
    sentiment: str        = Field(..., description="positive / neutral / negative")
    confidence: float     = Field(..., ge=0, le=1)
    probabilities: dict
    inference_time_ms: float
    detected_language: Optional[str] = None


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    version: str = "1.0.0"


class MetricsResponse(BaseModel):
    test_accuracy: float
    test_f1: float
    best_val_f1: float
    model_name: str
