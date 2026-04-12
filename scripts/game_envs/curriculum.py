"""Unified curriculum scheduler for all game environments.

Supports two hint-decay modes:

- ``"rollout"`` (liars_dice, leduc_poker, goof_spiel): hint probability decays
  linearly from ``initial_hint_prob`` to ``final_hint_prob`` based on total
  rollout count (after a warmup period).
- ``"optimizer_step"`` (gin_rummy): hint probability decays based on the
  trainer's ``global_step``. Passed explicitly to :meth:`get_hint_prob` since
  the scheduler has no direct access to trainer state.

Optionally warms up MCTS simulation count over optimizer steps (gin_rummy).
When ``initial_mcts_sims`` is ``None``, :meth:`get_mcts_sims` returns ``None``
and the rollout engine leaves the game's default MCTS config unchanged.
"""

from __future__ import annotations


class CurriculumScheduler:
    """Progressive turn-limit, hint, and MCTS-simulation curriculum."""

    def __init__(
        self,
        initial_max_turn: int = 2,
        final_max_turn: int = 20,
        rollouts_per_stage: int = 1280,
        initial_hint_prob: float = 0.0,
        final_hint_prob: float = 0.0,
        warmup_rollouts: int = 128,
        hint_decay_mode: str = "rollout",
        hint_decay_optimizer_steps: int = 0,
        mcts_warmup_optimizer_steps: int = 0,
        initial_mcts_sims: int | None = None,
        final_mcts_sims: int | None = None,
    ):
        if hint_decay_mode not in ("rollout", "optimizer_step"):
            raise ValueError(
                f"hint_decay_mode must be 'rollout' or 'optimizer_step', "
                f"got {hint_decay_mode!r}"
            )

        self.initial_max_turn = initial_max_turn
        self.final_max_turn = final_max_turn
        self.rollouts_per_stage = rollouts_per_stage
        self.initial_hint_prob = initial_hint_prob
        self.final_hint_prob = final_hint_prob
        self.warmup_rollouts = warmup_rollouts

        self.hint_decay_mode = hint_decay_mode
        self.hint_decay_optimizer_steps = hint_decay_optimizer_steps

        self.mcts_warmup_optimizer_steps = max(mcts_warmup_optimizer_steps, 0)
        self.initial_mcts_sims = initial_mcts_sims
        self.final_mcts_sims = final_mcts_sims

        self.total_rollouts = 0

    # ------------------------------------------------------------------ Turn
    def get_max_turn(self) -> int:
        """Current turn limit based on staged rollout progression."""
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_max_turn
        adjusted = self.total_rollouts - self.warmup_rollouts
        stage = adjusted // max(self.rollouts_per_stage, 1)
        return min(self.initial_max_turn + stage, self.final_max_turn)

    # ----------------------------------------------------------- Hint prob
    def get_hint_prob(self, optimizer_step: int | None = None) -> float:
        """Current probability of inserting strategy hints into the system prompt."""
        if self.hint_decay_mode == "optimizer_step":
            step = 0 if optimizer_step is None else max(optimizer_step, 0)
            if self.hint_decay_optimizer_steps <= 0:
                return self.final_hint_prob
            progress = min(step / self.hint_decay_optimizer_steps, 1.0)
            current = self.initial_hint_prob - progress * (
                self.initial_hint_prob - self.final_hint_prob
            )
            return max(current, self.final_hint_prob)

        # "rollout" mode
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_hint_prob
        total_stages = max(self.final_max_turn - self.initial_max_turn, 1)
        total_decay_rollouts = max(total_stages * self.rollouts_per_stage, 1)
        adjusted = self.total_rollouts - self.warmup_rollouts
        progress = min(adjusted / total_decay_rollouts, 1.0)
        current = self.initial_hint_prob - progress * (
            self.initial_hint_prob - self.final_hint_prob
        )
        return max(current, self.final_hint_prob)

    # ----------------------------------------------------------- MCTS sims
    def get_mcts_sims(self, optimizer_step: int | None = None) -> int | None:
        """Current MCTS simulation count, or ``None`` if the game doesn't vary it."""
        if self.initial_mcts_sims is None or self.final_mcts_sims is None:
            return None
        if self.mcts_warmup_optimizer_steps <= 0:
            return int(self.final_mcts_sims)
        step = 0 if optimizer_step is None else max(optimizer_step, 0)
        progress = min(step / self.mcts_warmup_optimizer_steps, 1.0)
        return int(
            self.initial_mcts_sims
            + progress * (self.final_mcts_sims - self.initial_mcts_sims)
        )

    # ------------------------------------------------------------------ Step
    def step(self, num_rollouts: int = 1) -> None:
        self.total_rollouts += num_rollouts

    def get_status(self, optimizer_step: int | None = None) -> dict:
        return {
            "total_rollouts": self.total_rollouts,
            "max_turn": self.get_max_turn(),
            "hint_prob": self.get_hint_prob(optimizer_step),
            "mcts_sims": self.get_mcts_sims(optimizer_step),
        }
