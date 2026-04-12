# Adding a New Game to `game_envs`

## Quick Start

1. Create `scripts/game_envs/games/<your_game>.py`
2. Register it in `scripts/game_envs/games/__init__.py`
3. Add its task ID range to `scripts/game_envs/shared.py`

That's it. The rollout engine, curriculum, trace logging, and token bookkeeping are handled automatically.

---

## Step 1: Implement `GameEnvironment`

Create a file in `games/` that subclasses `GameEnvironment`. You must implement **9 required methods** and can optionally override **7 hooks**.

```python
# scripts/game_envs/games/my_game.py

from game_envs.base import CurriculumDefaults, GameConfig, GameEnvironment
from game_envs.shared import clamp, remove_reasoning_tags

class MyGameEnvironment(GameEnvironment):

    # --- Required ---

    def get_config(self) -> GameConfig:
        return GameConfig(
            game_name="my_game",
            selected_game="my_game",          # key in GAME_TO_TASK_ID_RANGE
            mcts_config={"opponent": "mcts", "mcts_max_simulations": 50},
            invalid_action_penalty=0.10,       # per-invalid-action shaping penalty
            shaping_reward_clip=0.25,          # max |accumulated shaping|
            terminal_reward_clip=1.00,         # clamp on extract_terminal_reward
        )

    def get_curriculum_defaults(self) -> CurriculumDefaults:
        return CurriculumDefaults(
            initial_max_turn=2,                # turns at start of training
            final_max_turn=20,                 # turns at end of training
            rollouts_per_stage=1280,           # rollouts before incrementing max_turn
            initial_hint_prob=0.5,             # probability of strategy hints
            final_hint_prob=0.0,
            warmup_rollouts=128,               # rollouts at initial_max_turn before ramping
            hint_decay_mode="rollout",         # "rollout" or "optimizer_step"
        )

    def get_system_prompt(self, use_hints: bool) -> str:
        prompt = "You are playing my_game.\n\n# Rules\n..."
        if use_hints:
            prompt += "\n# Strategy Tips\n..."
        return prompt

    def extract_and_format_observation(self, raw_observation: str) -> str:
        # Transform server text into model-facing format.
        # Must include a "Legal Actions:" block for the default parser.
        return raw_observation or ""

    def extract_state_features(self, observation: str) -> dict:
        # Return game-specific features used by compute_step_shaping_reward.
        return {"some_feature": 0}

    def parse_action_id(self, completion_text: str, legal_action_map: dict[str, str]) -> str:
        # Extract action ID from model output. Return "" on failure.
        cleaned = remove_reasoning_tags(completion_text or "")
        for num in re.findall(r"-?\d+", cleaned):
            if num in legal_action_map:
                return num
        return ""

    def select_fallback_action(self, legal_action_map: dict[str, str], state_features: dict) -> str:
        # Called when parse_action_id fails. Pick a safe default.
        return sorted(legal_action_map.keys(), key=lambda x: int(x))[0]

    def compute_step_shaping_reward(self, state_features, action_id, action_label,
                                     legal_action_map, episode_state=None, mode="full") -> float:
        # Per-step reward shaping. Return 0.0 if not needed.
        return 0.0

    def extract_terminal_reward(self, step_block: dict, observation_text: str) -> float:
        # Extract final reward when episode ends.
        return float(step_block.get("reward", 0.0))
```

### Optional Hooks (override only if needed)

| Hook | Default | Override when... |
|------|---------|------------------|
| `extract_legal_action_map(obs)` | Parses `Legal Actions:\nID -> label` blocks | Your observation format differs |
| `build_reset_payload(game_id, mcts_config, seed)` | `{"task_id": game_id, "seed": seed, **mcts_config}` | You need extra reset fields |
| `on_episode_start(obs) -> Any` | Returns `None` | You need per-episode state (e.g. gin_rummy's `GameState` history) |
| `on_step_complete(obs, reward, done, is_invalid, action_id, episode_state) -> Any` | Returns `episode_state` unchanged | You update episode state each step |
| `compute_episode_reward(final_reward, accumulated_shaping, done, episode_state, step_rewards, mode)` | `final_reward + clamp(accumulated_shaping)` | You have custom reward logic (e.g. gin_rummy's deadwood improvement) |
| `pre_model_strategy_override(messages, obs, turn, max_turn, episode_state, mode) -> str\|None` | Returns `None` (always use model) | You want strategy forcing (e.g. goof_spiel plays expert moves, then trains on last turn) |
| `build_step_trace_record(**kwargs)` | Basic record with turn/action/reward/obs fields | You want extra trace fields |

---

## Step 2: Register the Game

In `scripts/game_envs/games/__init__.py`:

```python
from game_envs.games.my_game import MyGameEnvironment

register_game("my_game", MyGameEnvironment)
# Optional aliases: register_game("my_game", MyGameEnvironment, aliases=["mygame"])
```

## Step 3: Add Task ID Range

In `scripts/game_envs/shared.py`, add an entry to `GAME_TO_TASK_ID_RANGE`:

```python
GAME_TO_TASK_ID_RANGE = {
    ...
    "my_game": (800000000, 899999999),
}
```

## Step 4: Wire Up Training (if needed)

In `scripts/train_grpo_env.py`, if your game needs a custom `initial_max_turn` override, add it to `_INITIAL_MAX_TURN_OVERRIDES`:

```python
_INITIAL_MAX_TURN_OVERRIDES = {
    ...
    "my_game": 5,
}
```

No other changes to `train_grpo_env.py` are needed -- the registry handles dispatch automatically.

---

## Architecture Overview

```
scripts/game_envs/
  __init__.py          # Public API: get_game_environment(), get_rollout_funcs()
  base.py              # GameEnvironment ABC, GameConfig, CurriculumDefaults
  shared.py            # GAME_TO_TASK_ID_RANGE, remove_reasoning_tags, clamp, etc.
  curriculum.py        # CurriculumScheduler (turn ramp, hint decay, MCTS warmup)
  trace_logger.py      # EpisodeTraceLogger (JSONL per rank per game)
  env_client.py        # HTTP client for /reset and /step endpoints
  rollout_engine.py    # Generic rollout loop (token bookkeeping, action masking)
  registry.py          # Game registration and lazy instantiation
  games/
    __init__.py         # Imports + register_game() calls
    liars_dice.py       # Bid plausibility, challenge scoring
    leduc_poker.py      # Hand strength, fold/call/raise classification
    gin_rummy.py        # GameState, card utils, deadwood-based reward
    goof_spiel.py       # Strategy forcing, observation reformatting
```

### How It Works

1. `train_grpo_env.py` calls `get_game_environment("my_game")` to get a `GameEnvironment` instance
2. `get_rollout_funcs(game)` returns `(rollout_full, rollout_last, reward_func)` bound to that game
3. The rollout engine handles: env pool init, curriculum scheduling, parallel episode execution, token/action-mask bookkeeping, trace logging
4. Your game class only handles: prompts, observation parsing, action parsing, reward shaping

### Seed Policy

The engine randomizes seeds via `random.randint(0, 2**31 - 1)` on every `/reset`. Never use a fixed seed in `build_reset_payload` -- a fixed seed produces only ~9 unique games. The one-time health check during init is the only exception.

### Hint Decay Modes

- `"rollout"` (default): hint probability decays as total rollouts increase. Used by liars_dice, leduc_poker, goof_spiel.
- `"optimizer_step"`: hint probability decays based on the optimizer step counter. Set `hint_decay_optimizer_steps` to control the decay horizon. Used by gin_rummy.

### MCTS Warmup

Set `initial_mcts_sims`, `final_mcts_sims`, and `mcts_warmup_optimizer_steps` in `CurriculumDefaults` to ramp MCTS simulation count over training. Leave as `None` / `0` if unused.
