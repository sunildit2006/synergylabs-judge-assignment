"""
schema.py
---------
Structured output contract for the judge.
"""

from typing import Dict
from pydantic import BaseModel, Field

class CriterionScore(BaseModel):
    score: int = Field(..., ge=1, le=5)
    rationale: str = Field(...)

class PairwiseVerdict(BaseModel):
    criteria_breakdown: Dict[str, CriterionScore] = Field(...)
    winner: str = Field(...)
    rationale: str = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
