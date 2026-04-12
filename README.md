# GRPIO — autoresearch branch

> **EXPERIMENTAL** — This branch (`test/env-autoresearch`) contains an autonomous research loop that lets an LLM agent iteratively improve GRPO-Env training without human intervention between runs. It is under active development and should not be used in production or merged to `main` without review.

---

All of the code in this branch **HAS NOT BEEN TESTED**.

## What this is

GRPIO is a distributed LLM fine-tuning framework focused on **GRPO-Env** — reinforcement learning in interactive game environments. This branch adds an *autoresearch* workflow on top of that foundation: a structured loop where an LLM agent proposes, implements, trains, evaluates, and keeps or reverts experiments automatically.

The target metric is `avg_score` — win rate against an MCTS opponent (0.0–1.0, higher is better).

Currently active games: **Liar's Dice**, **Gin Rummy**, and **Leduc Poker**.

---

## TODO
- [ ] End-to-end test
- [ ] Real-time UI

---

## How the autoresearch loop works

Each iteration of the loop is a full train → eval cycle:

1. **Pick one idea** — reward shaping, observation formatting, prompt engineering, curriculum tuning, or hyperparameter adjustment (in that priority order).
2. **Modify only allowed files** — game environment file (`scripts/game_envs/games/<game>.py`), `scripts/grpo_env_config.py`, or eval knobs in `e2e.sh`. Everything else is read-only.
3. **Commit** the change.
4. **Train + eval**: `bash e2e.sh > train.log 2>&1` (evaluation runs automatically after training).
5. **Extract score**: `grep "Score:" train.log`
6. **Keep or revert**: improved score → keep commit; same or worse → `git reset --hard HEAD~1`.
7. **Log** every run to `results.tsv` (untracked).

The loop runs unattended until manually interrupted.

---

## Setup

```bash
# 1. Check out a named experiment branch (do not run autoresearch on main)
git checkout -b autoresearch/<tag>   # e.g. autoresearch/apr12-liars

# 2. Configure e2e.sh
#    - Set MODELS to the target model
#    - Confirm DATASET_TYPE matches the target game
#    - Set HOURS_TO_COMPLETE, HUGGINGFACE_TOKEN, WANDB_TOKEN

# 3. Initialize the results log (leave untracked)
echo -e "commit\tavg_score\thours_trained\tstatus\tdescription" > results.tsv

# 4. Run
bash e2e.sh > train.log 2>&1
```

GPU selection:

```bash
bash e2e.sh              # all GPUs
bash e2e.sh -g 0         # GPU 0 only
bash e2e.sh -g 0,1       # GPUs 0 and 1
```

---

## Checking results

```bash
# Check for crashes
grep "Training container.*finished with exit code" train.log

# Extract score
grep "Score:" train.log
# Output: Score: 137.5/200 (0.6875)
#                           ^^^^^^ this is avg_score
```

---

## Files an agent may edit

| File | What's allowed |
|------|----------------|
| `scripts/game_envs/games/<game>.py` | Reward shaping, system prompt, observation formatting, action parsing, curriculum defaults, MCTS config |
| `scripts/grpo_env_config.py` | Hyperparameters for the active model-size bucket |
| `e2e.sh` | `HOURS_TO_COMPLETE`, `EVAL_NUM_EVALS`, `EVAL_NUM_GPUS`, `RUN_EVAL_AFTER_TRAINING` |

Everything else — training pipeline, rollout engine, evaluation harness, Docker config — is **read-only**.

---

## Architecture

```
scripts/game_envs/
  base.py              # GameEnvironment ABC
  rollout_engine.py    # Generic rollout loop (off-limits)
  curriculum.py        # Turn ramp + hint decay
  games/
    liars_dice.py      # Bid plausibility, challenge scoring
    gin_rummy.py       # Deadwood-based reward, GameState history
    leduc_poker.py     # Hand strength, fold/call/raise
    goof_spiel.py      # Strategy forcing
```

See `scripts/game_envs/ADDING_GAMES.md` for the full architecture reference and instructions for adding new games.

---

## Important notes

- **Seeds**: `seed` in `/reset` payloads is always randomized via `random.randint(0, 2**31 - 1)`. A fixed seed produces only ~9 unique games.
- **`num_iterations`**: locked at `1` (fully on-policy). Do not increase without understanding the importance-sampling trade-off.
- **`steps_per_generation`**: defaults to `gradient_accumulation_steps`. Setting it lower wastes compute.
- **Noise floor**: ±0.005 on 200 eval games. Differences smaller than this are not meaningful.
