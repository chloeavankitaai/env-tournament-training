"""Game environment registry.

Games register themselves (lazily imported via ``get_game_environment``) and
callers resolve the public rollout/reward functions in one call::

    from game_envs import get_game_environment, get_rollout_funcs

    game = get_game_environment("liars_dice")
    rollout_full, rollout_last, reward_func = get_rollout_funcs(game)
"""

from __future__ import annotations

from typing import Callable

from game_envs.base import GameEnvironment
from game_envs.rollout_engine import make_rollout_functions


# Name -> factory. We use factories (not instances) so games are only
# constructed when actually requested, which keeps import time low.
_GAME_FACTORIES: dict[str, Callable[[], GameEnvironment]] = {}
_GAME_INSTANCES: dict[str, GameEnvironment] = {}


def register_game(
    name: str,
    factory: Callable[[], GameEnvironment],
    *,
    aliases: list[str] | None = None,
) -> None:
    """Register a game factory under ``name`` (and optional aliases)."""
    _GAME_FACTORIES[name] = factory
    for alias in aliases or []:
        _GAME_FACTORIES[alias] = factory


def get_game_environment(name: str) -> GameEnvironment:
    """Return a cached ``GameEnvironment`` instance for ``name``."""
    if name not in _GAME_INSTANCES:
        _ensure_games_imported()
        if name not in _GAME_FACTORIES:
            raise KeyError(
                f"Unknown game environment {name!r}. "
                f"Known: {sorted(_GAME_FACTORIES.keys())}"
            )
        _GAME_INSTANCES[name] = _GAME_FACTORIES[name]()
    return _GAME_INSTANCES[name]


def get_rollout_funcs(game: GameEnvironment):
    """Return ``(rollout_full, rollout_last, reward_func)`` bound to ``game``."""
    return make_rollout_functions(game)


def list_games() -> list[str]:
    """List all registered game names."""
    _ensure_games_imported()
    return sorted(_GAME_FACTORIES.keys())


def _ensure_games_imported() -> None:
    """Import the games package, which triggers game registration.

    Importing ``game_envs.games`` runs ``__init__.py`` which calls
    ``register_game`` for every concrete game.
    """
    from game_envs import games  # noqa: F401
