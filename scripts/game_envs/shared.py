"""Shared constants and utility helpers used across all game environments.

Everything in this module was previously copy-pasted (verbatim or near-verbatim)
into each ``*_environment_function.py`` file.
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Task ID ranges (shared with train_grpo_env.py for dataset construction)
# ---------------------------------------------------------------------------

GAME_TO_TASK_ID_RANGE: dict[str, tuple[int, int]] = {
    "goofspiel": (0, 99999999),
    "goof_spiel": (0, 99999999),       # alias used by environment_name config
    "liars_dice": (100000000, 199999999),
    "leduc_poker": (200000000, 299999999),
    "gin_rummy": (300000000, 399999999),
    "othello": (400000000, 499999999),
    "backgammon": (500000000, 599999999),
    "hex": (600000000, 699999999),
    "clobber": (700000000, 799999999),
}


# ---------------------------------------------------------------------------
# Reasoning-tag cleanup (every game stripped <think>/<reasoning>/... blocks)
# ---------------------------------------------------------------------------

REASONING_TAG_PAIRS: list[tuple[str, str]] = [
    ("think", "think"),
    ("thinking", "thinking"),
    ("reasoning", "reasoning"),
    ("thought", "thought"),
    ("reflection", "reflection"),
]


def remove_reasoning_tags(text: str) -> str:
    """Strip ``<think>...</think>`` style reasoning blocks from model output."""
    cleaned = text or ""
    for tag_name, close_name in REASONING_TAG_PAIRS:
        cleaned = re.sub(
            rf"<{tag_name}>.*?</{close_name}>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        close_tag = f"</{close_name}>"
        if close_tag in cleaned:
            cleaned = cleaned.split(close_tag)[-1]
        open_match = re.search(rf"<{tag_name}>", cleaned, flags=re.IGNORECASE)
        if open_match:
            cleaned = cleaned[: open_match.start()]
    cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def is_truthy_env(value: str | None) -> bool:
    """Return True if an environment variable string is a truthy flag."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce ``value`` to float, returning ``default`` on failure."""
    try:
        return float(value)
    except Exception:
        return default


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp ``value`` into ``[min_value, max_value]``."""
    return max(min_value, min(max_value, value))


# ---------------------------------------------------------------------------
# Legal-action parsing (shared between liars_dice and leduc_poker originally)
# ---------------------------------------------------------------------------

def extract_legal_action_map(observation: str) -> dict[str, str]:
    """Parse the ``Legal Actions:`` block into a mapping ``{action_id: label}``.

    Works for any observation where legal actions are listed as either
    ``123 -> label`` or just ``123`` (one per line) in a block terminated by
    ``Your choice`` or end of string.
    """
    if not observation:
        return {}
    match = re.search(
        r"Legal Actions:\s*\n(.*?)(?:\n\nYour choice|\nYour choice|\Z)",
        observation,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return {}

    block = match.group(1)
    mapping: dict[str, str] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "->" in line:
            left, right = line.split("->", 1)
            action_id = left.strip()
            label = right.strip()
        else:
            action_id = line.strip()
            label = action_id
        if re.fullmatch(r"-?\d+", action_id):
            mapping[action_id] = label
    return mapping


def extract_numeric_action_id(
    completion_text: str,
    legal_action_map: dict[str, str],
) -> str:
    """Numeric-only action extractor (common base for game-specific parsers).

    Strips reasoning tags, removes ``</s>`` end markers, splits on ``Action:``,
    and returns the first numeric token that exists in ``legal_action_map``.
    Returns ``""`` if nothing matches.
    """
    if not legal_action_map:
        return ""

    cleaned = remove_reasoning_tags(completion_text or "")
    if cleaned.endswith("</s>"):
        cleaned = cleaned[:-5]
    if "Action:" in cleaned:
        cleaned = cleaned.split("Action:")[-1].strip()

    for num in re.findall(r"-?\d+", cleaned):
        if num in legal_action_map:
            return num
    return ""


# Backward-compatibility aliases: the original files used underscored names.
_is_truthy_env = is_truthy_env
_safe_float = safe_float
_clamp = clamp
