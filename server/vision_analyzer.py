"""Optional model hooks for higher-level face estimates.

These functions intentionally return unknown until a real, validated model is
plugged in. Do not infer apparent gender, age, or expression from face presence
alone.
"""

from __future__ import annotations

from typing import Any


def estimate_apparent_gender(face_crop: Any) -> dict[str, float | str]:
    """Return apparent gender estimate from a future model hook."""
    _ = face_crop
    return {"label": "unknown", "confidence": 0.0}


def estimate_age_range(face_crop: Any) -> dict[str, float | str]:
    """Return estimated age range from a future model hook."""
    _ = face_crop
    return {"label": "unknown", "confidence": 0.0}


def estimate_expression(face_crop: Any) -> dict[str, float | str]:
    """Return expression estimate from a future model hook."""
    _ = face_crop
    return {"label": "unknown", "confidence": 0.0}
