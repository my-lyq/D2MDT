"""Rule-based department assignment for multi-agent consultation.

Scope constraints:
- No model training
- MIMIC-III and MIMIC-IV in-hospital mortality use case
- No prompt / retrieval / collaboration logic changes
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd

from utils.department_map import (
    BUCKET_WEIGHTS,
    DEPARTMENT_KEYWORDS,
    DEPARTMENT_CANDIDATES,
    DYNAMIC_DEPARTMENTS,
    FEATURE_TO_DEPARTMENT,
    FIXED_AGENT_ASSIGNMENT,
)


def _normalize_feature_name(name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _keyword_match_score(feature_name: str, keywords: Iterable[str]) -> float:
    name = _normalize_feature_name(feature_name)
    score = 0.0
    for kw in keywords:
        token = _normalize_feature_name(kw)
        if token in name or name in token:
            score += 1.0
    return score


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _dedupe(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        key = _normalize_feature_name(x)
        if key in seen:
            continue
        seen.add(key)
        out.append(str(x))
    return out


def _extract_numeric_block(raw_x, feature_names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    arr = np.asarray(raw_x)
    if arr.ndim != 2:
        return np.empty((0, 0), dtype=object), []

    # Keep consistent with consultation path where the first two columns are demo fields.
    if arr.shape[1] > 2:
        arr = arr[:, 2:]

    if not feature_names:
        return np.empty((0, 0), dtype=object), []

    if arr.shape[1] >= len(feature_names):
        block = arr[:, -len(feature_names):]
        names = list(feature_names)
    else:
        block = arr
        names = list(feature_names[: arr.shape[1]])

    return block, names


def extract_abnormal_numeric_features(
    raw_x,
    feature_names: Sequence[str],
    z_threshold: float = 1.0,
) -> List[str]:
    block, names = _extract_numeric_block(raw_x, feature_names)
    if block.size == 0:
        return []

    abnormalities: List[str] = []
    for j, feature_name in enumerate(names):
        col = pd.to_numeric(block[:, j], errors="coerce")
        finite = np.asarray(col[np.isfinite(col)], dtype=float)
        if finite.size < 2:
            continue
        last = float(finite[-1])
        median = float(np.median(finite))
        std = float(np.std(finite))
        z = abs(last - median) / (std + 1e-6)
        if z >= z_threshold:
            abnormalities.append(feature_name)

    return _dedupe(abnormalities)


def extract_important_feature_names(
    important_features: Sequence,
    top_k: int = 8,
) -> List[str]:
    scored: List[Tuple[float, str]] = []
    for item in important_features or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        name, payload = item
        if not isinstance(payload, dict):
            continue
        att = _safe_float(payload.get("attention"))
        if att is None:
            continue
        scored.append((abs(att), str(name)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return _dedupe([name for _, name in scored[:top_k]])


def _extract_risk_score(context_text: str) -> Optional[float]:
    pattern = r"mortality prediction risk[^\n]*?is\s*([0-9]*\.?[0-9]+)\s*out of 1\.0"
    m = re.search(pattern, context_text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _context_department_signals(context_text: str) -> Dict[str, float]:
    text = _normalize_feature_name(context_text)
    out = {dept: 0.0 for dept in DEPARTMENT_CANDIDATES}

    risk_score = _extract_risk_score(context_text)
    if risk_score is not None:
        if risk_score >= 0.6:
            out["critical_care_hemodynamic"] += 1.6
            out["respiratory_oxygenation"] += 0.8
        elif risk_score >= 0.3:
            out["critical_care_hemodynamic"] += 0.8
            out["metabolic_general"] += 0.4
        else:
            out["metabolic_general"] += 0.5

    if "top 30%" in text:
        out["critical_care_hemodynamic"] += 0.6
    if "icu" in text:
        out["critical_care_hemodynamic"] += 0.4

    return out


def score_departments(
    abnormal_numeric_features: Sequence[str],
    context_text: str,
    important_feature_names: Sequence[str],
) -> Dict[str, float]:
    scores = {dept: 0.0 for dept in DEPARTMENT_CANDIDATES}

    normalized_exact_map = {
        _normalize_feature_name(k): v for k, v in FEATURE_TO_DEPARTMENT.items()
    }

    buckets = {
        "abnormal_numeric": list(abnormal_numeric_features),
        "important_features": list(important_feature_names),
    }

    for bucket_name, features in buckets.items():
        weight = BUCKET_WEIGHTS[bucket_name]
        for feat in features:
            feat_norm = _normalize_feature_name(feat)

            exact_dept = normalized_exact_map.get(feat_norm)
            if exact_dept in DEPARTMENT_CANDIDATES:
                scores[exact_dept] += weight
                continue

            for dept in DEPARTMENT_CANDIDATES:
                kw_score = _keyword_match_score(feat, DEPARTMENT_KEYWORDS[dept])
                if kw_score > 0:
                    scores[dept] += weight * kw_score

    context_scores = _context_department_signals(context_text)
    for dept, s in context_scores.items():
        scores[dept] += BUCKET_WEIGHTS["context"] * s

    return scores


def assign_departments(
    abnormal_numeric_features: Sequence[str],
    context_text: str,
    important_feature_names: Sequence[str],
) -> Tuple[Dict[str, str], Dict[str, float]]:
    scores = score_departments(
        abnormal_numeric_features=abnormal_numeric_features,
        context_text=context_text,
        important_feature_names=important_feature_names,
    )

    ranked_dynamic = sorted(
        DYNAMIC_DEPARTMENTS,
        key=lambda d: (-scores.get(d, 0.0), d),
    )

    assignment = {
        "agent_1": FIXED_AGENT_ASSIGNMENT["agent_1"],
        "agent_2": ranked_dynamic[0],
        "agent_3": ranked_dynamic[1],
    }
    return assignment, scores


def build_assignment_signals(
    raw_x,
    feature_names: Sequence[str],
    hcontext: str,
    important_features: Sequence,
) -> Dict[str, object]:
    abnormal_numeric_features = extract_abnormal_numeric_features(raw_x=raw_x, feature_names=feature_names)
    important_feature_names = extract_important_feature_names(important_features=important_features)
    return {
        "abnormal_numeric_features": abnormal_numeric_features,
        "important_feature_names": important_feature_names,
        "important_features_empty": len(important_feature_names) == 0,
        "context_text": hcontext,
    }
