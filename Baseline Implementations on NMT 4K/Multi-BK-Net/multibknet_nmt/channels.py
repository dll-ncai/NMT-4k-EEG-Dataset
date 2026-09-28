from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

_CANONICAL_CASE = {
    "FP1": "Fp1",
    "FP2": "Fp2",
    "F7": "F7",
    "F3": "F3",
    "FZ": "Fz",
    "F4": "F4",
    "F8": "F8",
    "T3": "T3",
    "C3": "C3",
    "CZ": "Cz",
    "C4": "C4",
    "T4": "T4",
    "T5": "T5",
    "P3": "P3",
    "PZ": "Pz",
    "P4": "P4",
    "T6": "T6",
    "O1": "O1",
    "O2": "O2",
    "A1": "A1",
    "A2": "A2",
}

_MODERN_TO_LEGACY = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
    "M1": "A1",
    "M2": "A2",
}


def channel_token(name: str) -> str:
    token = str(name).strip().upper()
    token = token.replace("–", "-").replace("—", "-")
    token = re.sub(r"^EEG[\s:_-]*", "", token)
    token = re.sub(r"\s+", "", token)
    token = token.replace("_", "").replace(".", "")
    token = re.sub(r"-(REF|LE|AVG|AR)$", "", token)
    return token


def canonicalize_channel(name: str) -> str | None:
    token = channel_token(name)
    token = _MODERN_TO_LEGACY.get(token, token)
    return _CANONICAL_CASE.get(token)


def select_channel_indices(
    raw_channel_names: Iterable[str], required_channels: Iterable[str]
) -> tuple[list[int], dict[str, str]]:
    raw_channel_names = list(raw_channel_names)
    candidates: dict[str, list[int]] = defaultdict(list)
    for index, raw_name in enumerate(raw_channel_names):
        canonical = canonicalize_channel(raw_name)
        if canonical is not None:
            candidates[canonical].append(index)

    indices: list[int] = []
    mapping: dict[str, str] = {}
    missing: list[str] = []
    duplicates: dict[str, list[str]] = {}

    for required in required_channels:
        canonical_required = canonicalize_channel(required) or required
        matches = candidates.get(canonical_required, [])
        if not matches:
            missing.append(required)
            continue
        if len(matches) > 1:
            exact = [
                i
                for i in matches
                if channel_token(raw_channel_names[i]) == channel_token(required)
            ]
            if len(exact) == 1:
                chosen = exact[0]
            else:
                duplicates[required] = [raw_channel_names[i] for i in matches]
                continue
        else:
            chosen = matches[0]
        indices.append(chosen)
        mapping[required] = raw_channel_names[chosen]

    if missing or duplicates:
        parts: list[str] = []
        if missing:
            parts.append(f"missing channels: {', '.join(missing)}")
        if duplicates:
            duplicate_text = "; ".join(
                f"{key}={value}" for key, value in duplicates.items()
            )
            parts.append(f"ambiguous duplicate channels: {duplicate_text}")
        raise ValueError("; ".join(parts))

    return indices, mapping
