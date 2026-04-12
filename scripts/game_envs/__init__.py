"""Shared infrastructure for GRPO-Env game environment rollouts.

This package centralizes the rollout loop, curriculum scheduling, environment
client, and trace logging that were previously duplicated across the four
game-specific environment files. Individual games live under ``games/`` and
implement the :class:`GameEnvironment` ABC defined in :mod:`base`.

Public API for callers (e.g. ``train_grpo_env.py``)::

    from game_envs import get_game_environment, get_rollout_funcs

    game = get_game_environment("liars_dice")
    rollout_full, rollout_last, reward_func = get_rollout_funcs(game)
"""

from game_envs.base import GameConfig, CurriculumDefaults, GameEnvironment
from game_envs.curriculum import CurriculumScheduler
from game_envs.registry import (
    get_game_environment,
    get_rollout_funcs,
    register_game,
    list_games,
)
from game_envs.shared import (
    GAME_TO_TASK_ID_RANGE,
    REASONING_TAG_PAIRS,
    remove_reasoning_tags,
)

__all__ = [
    "GameConfig",
    "CurriculumDefaults",
    "GameEnvironment",
    "CurriculumScheduler",
    "get_game_environment",
    "get_rollout_funcs",
    "register_game",
    "list_games",
    "GAME_TO_TASK_ID_RANGE",
    "REASONING_TAG_PAIRS",
    "remove_reasoning_tags",
]
