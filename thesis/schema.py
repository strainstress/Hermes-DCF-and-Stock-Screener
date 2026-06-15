"""JSON schema and validation for AI-generated analyst reports.

Defines the exact output format the LLM must produce.
Validates and retries on schema violations.
"""

import json
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from loguru import logger


# ── Pydantic Models ────────────────────────────────────────────


class QualitativeScores(BaseModel):
    moat: int = Field(ge=0, le=10, description="Competitive moat strength (0-10)")
    management: int = Field(ge=0, le=10, description="Management quality (0-10)")
    capital_allocation: int = Field(ge=0, le=10, description="Capital allocation discipline (0-10)")
    industry_position: int = Field(ge=0, le=10, description="Industry position / market power (0-10)")


class AnalystReport(BaseModel):
    """The exact schema expected from the LLM."""

    tldr: str = Field(description="2-3 sentence summary of the investment thesis")
    bull_case: list[str] = Field(min_length=1, max_length=5, description="Bull case points")
    bear_case: list[str] = Field(min_length=1, max_length=5, description="Bear case / risk points")
    key_metrics_to_watch: list[str] = Field(min_length=1, max_length=5, description="Key metrics to monitor")
    qualitative_scores: QualitativeScores
    peer_comparison: str = Field(description="1-2 paragraph comparison vs closest peers")
    verdict: Literal["Strong Buy", "Buy", "Watch", "Pass", "Sell"] = Field(description="Final verdict")
    confidence: int = Field(ge=0, le=100, description="Confidence in the verdict (0-100)")
    sources_cited: list[str] = Field(min_length=0, max_length=10, description="Specific sources cited (10-K pages, transcript quotes, etc.)")


# ── Validation ────────────────────────────────────────────────


def validate_report(data: dict | str) -> tuple[bool, AnalystReport | None, str]:
    """Validate a raw LLM output against the AnalystReport schema.

    Args:
        data: Raw dict or JSON string from the LLM.

    Returns:
        (is_valid, parsed_model_or_None, error_message)
    """
    if isinstance(data, str):
        try:
            # Handle potential markdown code fences
            cleaned = _extract_json(data)
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            return False, None, f"JSON parse error: {e}"

    try:
        report = AnalystReport.model_validate(data)
        return True, report, ""
    except ValidationError as e:
        return False, None, str(e)


def _extract_json(text: str) -> str:
    """Extract JSON from text that may be wrapped in markdown code fences."""
    # Try to find JSON between ```json ... ``` fences
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        return text[start:end].strip()
    # Try to find first { and last }
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        return text[brace_start:brace_end + 1]
    return text
