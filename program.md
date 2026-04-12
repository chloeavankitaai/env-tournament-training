# autoresearch — GRPIO

This is an experiment to have the LLM autonomously research GRPO-Env training for game-playing agents.

## Setup

Work with the user to complete setup before starting the loop:

1. **Agree on a run tag**: propose a tag based on today's date and the target game (e.g. `apr11-gin` or `apr11-liars`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current branch.
3. **Read the in-scope files** (read all of these for full context before touching anything):
  - `CLAUDE.md` — project overview, conventions, architecture.
  - `e2e.sh` — the training + eval launcher. Note the active model (`MODELS`), game (`DATASET_TYPE`), `HOURS_TO_COMPLETE`, and eval config (`RUN_EVAL_AFTER_TRAINING`, `EVAL_NUM_EVALS`, `EVAL_NUM_GPUS`). Evaluation runs automatically after training completes.
  - `scripts/grpo_env_config.py` — hyperparameter configs per model-size bucket.
  - The **active game's environment file** — all game-specific logic lives in `scripts/game_envs/games/<game>.py` (e.g. `gin_rummy.py`, `liars_dice.py`). This contains reward shaping constants, MCTS opponent config, strategy tips, observation formatting, and action parsing.
  - `scripts/game_envs/ADDING_GAMES.md` — reference for the game environment architecture and how to add/modify games.
  - `manual_environment_eval.py` — the evaluation harness. This is the ground truth; do not modify it. It is invoked automatically by `e2e.sh` after training (via env vars), so you never need to edit it or run it manually.
4. **Confirm the active experiment target** with the user: which model (e.g. `mistralai/Mistral-7B-Instruct-v0.3`) and which game (e.g. `gin_rummy`). These determine which config bucket and which environment function file you edit.
5. **Initialize results.tsv**: create it with just the header row (see Logging section below). Leave it untracked by git.
6. **Confirm and go**: confirm setup looks good with the user, then kick off.

---

## What you CAN modify

These are the only files you edit. Everything else is read-only.


| File | What's fair game |
| ---- | ---------------- |
| `scripts/grpo_env_config.py` | Hyperparameters in the active model-size bucket: `lr`, `beta`, `num_generations`, `batch_size`, `gradient_accumulation_steps`, `rollouts_per_stage`, `rollout_warmup_rollouts`, `mcts_warmup_optimizer_steps`, `initial_max_turn`, `vllm_gpu_memory_utilization`. |
| `scripts/game_envs/games/<game>.py` | Everything inside the game class — reward shaping constants, system prompt text, strategy tips, MCTS config, observation formatting, action parsing, curriculum defaults, episode-state hooks, step/episode reward logic. You may freely add helpers, restructure internals, or redesign reward shaping from scratch. The class must continue to subclass `GameEnvironment` from `game_envs.base`. |
| `e2e.sh` | `HOURS_TO_COMPLETE`, `EVAL_NUM_EVALS`, `EVAL_NUM_GPUS`, `RUN_EVAL_AFTER_TRAINING`. Do not change `MODELS`, `DATASET_TYPE`, or any Docker/upload config. |


## What you CANNOT modify

- `scripts/train_grpo_env.py` — core training pipeline. Off-limits.
- `scripts/game_envs/rollout_engine.py`, `scripts/game_envs/base.py`, `scripts/game_envs/registry.py` — shared rollout infrastructure. Off-limits.
- `manual_environment_eval.py` — the evaluation harness and metric. Off-limits.
- `scripts/prepare.py`, `scripts/model_utility.py`, `scripts/lrs_lookup.py`, `scripts/customized_trainer.py` — infrastructure. Off-limits.
- Docker configuration, upload scripts, or anything that changes how evaluation results are collected.

---

## The metric

**The goal: maximize `avg_score`** (average win rate against the MCTS opponent, from `manual_environment_eval.py`).

- `avg_score` is between 0.0 and 1.0. A random agent scores ~0.5 on symmetric games; anything above that is real signal.
- Higher is better. This is the only number that counts.

---

## Running a training experiment

Each experiment is a full train → eval cycle. **Evaluation runs automatically after training** — a single command does both.

**Step 1 — Train + Eval:**

```bash
bash e2e.sh > train.log 2>&1
```

This builds Docker images, starts environment servers, downloads model/dataset, runs training, and then **automatically evaluates** the latest checkpoint against the MCTS opponent. Logs are streamed and saved to `training_logs/`.

Training takes the duration set in `HOURS_TO_COMPLETE` (typically 3 hours). Evaluation adds ~30–60 minutes depending on `EVAL_NUM_EVALS`. Do NOT run this in the foreground if you want to do other work — redirect to a log file as shown.

The post-training eval is controlled by three variables near the top of `e2e.sh`:
- `RUN_EVAL_AFTER_TRAINING=1` — set to `0` to skip evaluation entirely
- `EVAL_NUM_EVALS=200` — number of evaluation games (keep at 200 for consistent results)
- `EVAL_NUM_GPUS=2` — GPUs for SGLang inference during eval

The game is auto-detected from `DATASET_TYPE` — no need to set it manually.

**Step 2 — Check for crashes:**

```bash
grep -i "error\|traceback\|exception\|killed\|oom" train.log | tail -20
grep "Training container.*finished with exit code" train.log
```

Exit code 0 means success. Non-zero or no exit code line means crash.

If a crash: read `tail -n 100 train.log` for the traceback. Fix the obvious bug and re-run. If the idea is fundamentally broken, log it as `crash` and move on.

**Step 3 — Extract the eval score:**

```bash
grep "Score:" train.log
```

Output looks like: `Score: 137.5/200 (0.6875)`

The `avg_score` is the number in parentheses (e.g. `0.6875`).

Full eval results are also saved to `eval_results/<game>/<model>/` and eval logs to `training_logs/eval_<model>_<timestamp>.log`.

**Step 4 — (Optional) Run eval manually:**

If you need to re-evaluate a checkpoint (e.g. with different settings) without re-training:

```bash
BASE_MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3" \
LOCAL_LORA_PATH="outputs/1/<repo_name>/checkpoint-150" \
GAME_TO_EVAL="liars_dice" \
NUM_EVALS=200 \
NUM_GPUS=2 \
python manual_environment_eval.py > eval.log 2>&1
```

All eval config vars (`GAME_TO_EVAL`, `OPPONENT_TYPE`, `MCTS_MAX_SIMULATIONS`, `NUM_EVALS`, `TEMPERATURE`, etc.) accept env var overrides. When `BASE_MODEL_NAME` is set as an env var, the script skips its internal `model_list` and evaluates just the specified model.

---

## Logging results

Record every run to `results.tsv` (tab-separated — do NOT use commas). Leave it untracked by git.

Header and columns:

```
commit	avg_score	hours_trained	status	description
```

1. `commit` — short git hash (7 chars) of the code change
2. `avg_score` — float from eval (e.g. `0.6875`); use `0.0000` for crashes
3. `hours_trained` — value of `HOURS_TO_COMPLETE` used (e.g. `3`)
4. `status` — `keep`, `discard`, or `crash`
5. `description` — short description of what this experiment changed (tabs in description will break parsing — use spaces)

Example:

```
commit	avg_score	hours_trained	status	description
a1b2c3d	0.5100	3	keep	baseline
b2c3d4e	0.5450	3	keep	increase beta 0.01->0.04 to reduce entropy collapse
c3d4e5f	0.5200	3	discard	lower lr 1e-5->5e-6 (worse performance)
d4e5f6g	0.0000	3	crash	doubled num_generations OOM
e5f6g7h	0.5600	3	keep	add bid plausibility bonus 0.02->0.05 + reduce invalid penalty
```

---

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr11-gin`).

**LOOP FOREVER:**

1. **Check git state**: confirm the current branch and commit.
2. **Pick an experimental idea**: one change at a time. Mixing multiple ideas makes it impossible to attribute causality.

   **The environment function is your primary lever.** Hyperparameter tuning has limited ceiling; the environment function determines what signal the model receives every step. Spend the majority of your experiments here before touching config knobs. Ideas, in order of expected impact:

   **Tier 1 — Game environment file (highest leverage, explore first):**
   All of these live in `scripts/game_envs/games/<game>.py`:
   - Reward shaping: tune constants at the top of the file, or redesign `compute_step_shaping_reward()` / `compute_episode_reward()` — intermediate bonuses, shaped terminal rewards, win-margin scaling, partial credit for near-legal moves
   - Observation formatting: rewrite `extract_and_format_observation()` — what information is included, how it's structured, what is elided
   - Prompt engineering: rewrite `SYSTEM_PROMPT_BASE`, `STRATEGY_TIPS`, change `get_system_prompt()` framing
   - Curriculum defaults: `initial_max_turn`, `final_max_turn`, hint probabilities in `get_curriculum_defaults()`
   - MCTS opponent strength: `mcts_max_simulations`, `mcts_num_rollouts` in `get_config().mcts_config` — weaker opponents give denser learning signal early
   - Episode-state hooks: `on_episode_start()`, `on_step_complete()` for tracking game-specific state across turns

   **Tier 2 — Hyperparameters (tune after environment is well-designed):**
   - GRPO knobs: `beta`, `num_generations`, `rollouts_per_stage`
   - Learning rate: `lr`, combined with `HOURS_TO_COMPLETE`
3. **Modify the file(s)**, then `git commit -m "brief description of change"`.
4. **Train + Eval**: `bash e2e.sh > train.log 2>&1` (evaluation runs automatically after training)
5. **Check for crash**: `grep "finished with exit code" train.log`
6. **Extract score**: `grep "Score:" train.log`
7. **Record** in `results.tsv`.
8. **Keep or discard**:
  - `avg_score` improved → keep the commit, advance the branch.
  - `avg_score` same or worse → `git reset --hard HEAD~1` to revert.
  - Crash → fix or skip; `git reset --hard HEAD~1` if the idea was unsalvageable.

**On ties**: if new `avg_score` equals baseline within ±0.005 (noise floor given 200 evals), prefer the simpler code. Keep a simplification, discard unnecessary complexity.

**Simplicity criterion**: same as karpathy/autoresearch — a small improvement that adds ugly hacks isn't worth it. Removing something for equal performance is a win.

**Timeout**: training has its own built-in timeout (`HOURS_TO_COMPLETE`). If the Docker container hangs beyond `HOURS_TO_COMPLETE + 30min`, kill it: `docker stop $(docker ps -q --filter name=grpo-text-trainer)`.

**NEVER STOP**: once the loop has begun, do NOT pause to ask the human if you should continue. The human may be asleep. Run until manually interrupted. If you run out of obvious ideas, consult `knowledge/` for recent RL research insights and apply them. Read prior results in `results.tsv` to find near-misses worth revisiting or combining.

---

## Key implementation details to keep in mind

- **Seed policy**: the engine randomizes seeds via `random.randint(0, 2**31 - 1)` on every `/reset`. If you override `build_reset_payload()`, keep this behavior. A fixed seed produces ~9 unique games.
- **`num_iterations`**: currently locked at `1` (fully on-policy). Do not increase without understanding the importance-sampling trade-off.
- **`steps_per_generation`**: defaults to `gradient_accumulation_steps`. Setting it lower is wasteful.
- **GPU memory**: `vllm_gpu_memory_utilization` is a soft constraint. Small increases are fine for meaningful gains; OOMs are not.
- **Game file location**: all game-specific logic lives in `scripts/game_envs/games/<game>.py`. The game name in `get_config().game_name` must match the `environment_name` used in `e2e.sh` / `DATASET_TYPE`. See `scripts/game_envs/ADDING_GAMES.md` for the full architecture reference.

