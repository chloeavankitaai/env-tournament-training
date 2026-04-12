"""Concrete game implementations.

Importing this package registers every game with the registry. Adding a new
game means creating ``games/<name>.py`` that subclasses ``GameEnvironment``
and appending a ``register_game(...)`` line below.
"""

from game_envs.registry import register_game

from game_envs.games.liars_dice import LiarsDiceEnvironment
from game_envs.games.leduc_poker import LeducPokerEnvironment
from game_envs.games.gin_rummy import GinRummyEnvironment
from game_envs.games.goof_spiel import GoofSpielEnvironment


register_game("liars_dice", LiarsDiceEnvironment)
register_game("leduc_poker", LeducPokerEnvironment)
register_game("gin_rummy", GinRummyEnvironment)
# goof_spiel uses two names historically (goof_spiel in train_grpo_env,
# goofspiel in GAME_TO_TASK_ID_RANGE), so register both.
register_game("goof_spiel", GoofSpielEnvironment, aliases=["goofspiel"])


__all__ = [
    "LiarsDiceEnvironment",
    "LeducPokerEnvironment",
    "GinRummyEnvironment",
    "GoofSpielEnvironment",
]
