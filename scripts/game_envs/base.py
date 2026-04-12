"""Base classes for game-specific environment implementations.

Every concrete game implements :class:`GameEnvironment`. Required methods
define game-specific behavior (prompts, action parsing, reward shaping);
optional hooks provide sensible defaults that most games can use as-is.

The rollout engine (:mod:`rollout_engine`) is generic over GameEnvironment,
so adding a new game does not require touching the shared loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from game_envs.shared import clamp, extract_legal_action_map, extract_numeric_action_id


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GameConfig:
    """Immutable per-game configuration.

    Defaults match the current liars_dice/leduc_poker values.
    """

    game_name: str                                    # e.g. "liars_dice"
    selected_game: str                                # key in GAME_TO_TASK_ID_RANGE
    mcts_config: dict[str, Any] = field(default_factory=dict)
    request_timeout_seconds: int = 2400
    init_timeout_seconds: int = 300
    max_episode_tokens: int = 16384
    max_prompt_len: int = 16384 - 512
    invalid_action_penalty: float = 0.10
    shaping_reward_clip: float = 0.25
    terminal_reward_clip: float = 1.00


@dataclass
class CurriculumDefaults:
    """Default curriculum scheduler parameters for a game.

    These are used to construct the :class:`CurriculumScheduler`; they may be
    overridden at runtime via environment variables or trainer args.
    """

    initial_max_turn: int = 2
    final_max_turn: int = 20
    rollouts_per_stage: int = 1280
    initial_hint_prob: float = 0.0
    final_hint_prob: float = 0.0
    warmup_rollouts: int = 128

    # Hint decay: either rollout-based (default) or optimizer-step-based
    hint_decay_mode: str = "rollout"                  # "rollout" | "optimizer_step"
    hint_decay_optimizer_steps: int = 0

    # MCTS warmup (optional — gin_rummy is the only current user)
    mcts_warmup_optimizer_steps: int = 0
    initial_mcts_sims: int | None = None
    final_mcts_sims: int | None = None


# ---------------------------------------------------------------------------
# GameEnvironment ABC
# ---------------------------------------------------------------------------

class GameEnvironment(ABC):
    """Interface implemented by each game.

    The rollout engine calls these methods to construct prompts, parse
    actions, compute rewards, and log traces. Optional hooks have working
    defaults and only need to be overridden for unusual games (goof_spiel's
    strategy forcing, gin_rummy's episode-level reward calculator).
    """

    # ----------------------------------------------------- Required methods

    @abstractmethod
    def get_config(self) -> GameConfig:
        """Return static configuration for this game."""

    @abstractmethod
    def get_curriculum_defaults(self) -> CurriculumDefaults:
        """Return default curriculum parameters."""

    @abstractmethod
    def get_system_prompt(self, use_hints: bool) -> str:
        """Build the system prompt (game rules + optional strategy tips)."""

    @abstractmethod
    def extract_and_format_observation(self, raw_observation: str) -> str:
        """Transform a raw server observation into the model-facing format."""

    @abstractmethod
    def extract_state_features(self, observation: str) -> dict:
        """Extract game-specific state features used by reward shaping."""

    @abstractmethod
    def parse_action_id(
        self,
        completion_text: str,
        legal_action_map: dict[str, str],
    ) -> str:
        """Parse model output into an action id. Return ``""`` on failure."""

    @abstractmethod
    def select_fallback_action(
        self,
        legal_action_map: dict[str, str],
        state_features: dict,
    ) -> str:
        """Choose a reasonable default when parsing fails."""

    @abstractmethod
    def compute_step_shaping_reward(
        self,
        state_features: dict,
        action_id: str,
        action_label: str,
        legal_action_map: dict[str, str],
        episode_state: Any = None,
        mode: str = "full",
    ) -> float:
        """Per-step shaping reward (accumulated and clipped by the engine).

        ``mode`` is ``"full"`` or ``"last"`` depending on which rollout
        variant the engine is running. ``episode_state`` is the value
        currently held by the engine (may be ``None``).
        """

    @abstractmethod
    def extract_terminal_reward(
        self,
        step_block: dict,
        observation_text: str,
    ) -> float:
        """Compute the final episode reward from the terminal step."""

    # ----------------------------------------------------- Optional hooks

    def extract_legal_action_map(self, observation: str) -> dict[str, str]:
        """Parse the ``Legal Actions:`` block. Default: shared regex parser."""
        return extract_legal_action_map(observation)

    def build_reset_payload(
        self,
        game_id: int,
        mcts_config: dict,
        seed: int,
    ) -> dict:
        """Construct the /reset payload. Override for custom fields."""
        return {"task_id": game_id, "seed": seed, **mcts_config}

    def on_episode_start(self, observation: str) -> Any:
        """Hook called after reset. Return game-specific episode state."""
        return None

    def on_step_complete(
        self,
        observation: str,
        step_reward: float,
        done: bool,
        is_invalid: bool,
        action_id: str,
        episode_state: Any,
    ) -> Any:
        """Hook called after each environment step. Update episode state."""
        return episode_state

    def compute_episode_reward(
        self,
        final_reward: float,
        accumulated_shaping: float,
        done: bool,
        episode_state: Any,
        step_rewards: list[float],
        mode: str = "full",
    ) -> float:
        """Combine terminal + shaping into the training reward.

        Default: clamp accumulated shaping, add terminal reward.
        Override for games with more elaborate logic (gin_rummy, goof_spiel).
        """
        config = self.get_config()
        clipped = clamp(
            accumulated_shaping,
            -config.shaping_reward_clip,
            config.shaping_reward_clip,
        )
        return final_reward + clipped

    def pre_model_strategy_override(
        self,
        messages: list[dict],
        observation: str,
        turn_number: int,
        max_turn: int,
        episode_state: Any,
        mode: str = "full",
    ) -> str | None:
        """Optionally bypass model generation for strategy forcing.

        If this returns a non-``None`` action id string, the engine uses it
        directly instead of calling the model. Used by goof_spiel to play
        expert moves during the strategy-forcing phase.

        Default: always use the model.
        """
        return None

    def build_step_trace_record(
        self,
        *,
        turn: int,
        completion_text: str,
        action_id: str,
        action_label: str,
        state_features: dict,
        shaping_reward: float,
        observation_before: str,
        observation_after: str,
        step_reward: float,
        done: bool,
        invalid_or_noop: bool,
        parse_failed: bool,
        trace_logger,
    ) -> dict:
        """Build the per-step trace record. Override to add custom fields."""
        clip = trace_logger.clip_text if trace_logger else (lambda s: s)
        return {
            "turn": turn,
            "assistant_text": clip(completion_text),
            "parsed_action": action_id,
            "action_label": action_label,
            "observation_before_action": clip(observation_before),
            "observation_after_action": clip(observation_after),
            "step_reward": float(step_reward),
            "shaping_reward": float(shaping_reward),
            "done": bool(done),
            "invalid_or_noop": bool(invalid_or_noop),
            "parse_failed": bool(parse_failed),
        }


# ---------------------------------------------------------------------------
# Convenience: default numeric parser exposed for games that don't need more
# ---------------------------------------------------------------------------

def default_numeric_action_parser(
    completion_text: str,
    legal_action_map: dict[str, str],
) -> str:
    """Simple numeric action parser usable by most games."""
    return extract_numeric_action_id(completion_text, legal_action_map)
