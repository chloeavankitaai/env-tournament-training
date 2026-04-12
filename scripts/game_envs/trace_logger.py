"""Thread-safe JSONL episode tracer (parameterized by game name)."""

from __future__ import annotations

import json
import os
import random
from threading import Lock


class EpisodeTraceLogger:
    """Append-only JSONL writer for per-episode rollout traces.

    Previously duplicated verbatim in liars_dice and leduc_poker with the
    log filename hardcoded. Now parameterized by ``game_name``.
    """

    def __init__(self, trace_dir: str, rank: int, game_name: str):
        self.trace_dir = trace_dir
        self.rank = rank
        self.game_name = game_name
        self._lock = Lock()
        self.log_path = os.path.join(
            self.trace_dir, f"{game_name}_episode_traces_rank{rank}.jsonl"
        )
        self.max_text_chars = int(
            os.environ.get("EPISODE_TRACE_MAX_TEXT_CHARS", "4000")
        )
        self.sample_rate = float(os.environ.get("EPISODE_TRACE_SAMPLE_RATE", "1.0"))

        os.makedirs(self.trace_dir, exist_ok=True)
        print(f"[EPISODE_TRACE] Writing traces to {self.log_path}")

    def should_log(self) -> bool:
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        return random.random() <= self.sample_rate

    def clip_text(self, text: str) -> str:
        if not text:
            return ""
        if len(text) <= self.max_text_chars:
            return text
        return (
            text[: self.max_text_chars]
            + f"... [truncated {len(text) - self.max_text_chars} chars]"
        )

    def log_episode(self, payload: dict) -> None:
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")
