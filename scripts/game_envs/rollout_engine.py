"""Generic rollout engine shared by all game environments.

The engine owns:
- Per-game lazy initialization (env pool, thread pool, semaphore, curriculum,
  trace logger). Stored in ``_ROLLOUT_STATES`` keyed by game name so multiple
  games can coexist in the same process.
- The main interaction loop: reset -> (generate -> parse -> step -> reward)* ->
  aggregate. All token/action-mask bookkeeping lives here; games don't see it.
- Parallel execution via a per-game ``ThreadPoolExecutor``.
- Trace logging and batch statistics.

Game-specific behavior is delegated to the :class:`GameEnvironment` instance
passed in. See :mod:`base` for the full interface.
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from threading import Semaphore
from typing import Any

from game_envs.base import GameEnvironment
from game_envs.curriculum import CurriculumScheduler
from game_envs.env_client import build_env_pool, reset_environment, step_environment
from game_envs.shared import GAME_TO_TASK_ID_RANGE, clamp, is_truthy_env
from game_envs.trace_logger import EpisodeTraceLogger


# Per-game rollout state. Initialized once per process on first call.
_ROLLOUT_STATES: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def _ensure_initialized(game: GameEnvironment, trainer) -> dict:
    """Lazily build the rollout state for a game on first call."""
    config = game.get_config()
    game_name = config.game_name

    state = _ROLLOUT_STATES.get(game_name)
    if state is not None and state.get("initialized"):
        return state

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
    server_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]
    if not server_urls:
        raise RuntimeError("ENVIRONMENT_SERVER_URLS is empty")

    init_task_id = GAME_TO_TASK_ID_RANGE[config.selected_game][0]
    env_pool = build_env_pool(
        server_urls=server_urls,
        init_task_id=init_task_id,
        mcts_config=config.mcts_config,
        init_timeout_seconds=config.init_timeout_seconds,
    )

    # Allow trainer args and env vars to override curriculum defaults.
    defaults = game.get_curriculum_defaults()
    game_upper = game_name.upper()

    initial_max_turn = int(
        getattr(trainer.args, "initial_max_turn", None) or defaults.initial_max_turn
    )
    final_max_turn = int(
        os.environ.get(f"{game_upper}_FINAL_MAX_TURN", defaults.final_max_turn)
    )
    rollouts_per_stage = int(
        getattr(trainer.args, "rollouts_per_stage", None) or defaults.rollouts_per_stage
    )
    initial_hint_prob = float(
        os.environ.get(f"{game_upper}_INITIAL_HINT_PROB", defaults.initial_hint_prob)
    )
    final_hint_prob = float(
        os.environ.get(f"{game_upper}_FINAL_HINT_PROB", defaults.final_hint_prob)
    )

    warmup_rollouts_arg = getattr(trainer.args, "rollout_warmup_rollouts", None)
    warmup_rollouts = (
        int(warmup_rollouts_arg)
        if warmup_rollouts_arg is not None
        else defaults.warmup_rollouts
    )

    mcts_warmup_optimizer_steps = getattr(
        trainer.args, "mcts_warmup_optimizer_steps", None
    )
    if mcts_warmup_optimizer_steps is None:
        mcts_warmup_optimizer_steps = defaults.mcts_warmup_optimizer_steps

    curriculum = CurriculumScheduler(
        initial_max_turn=initial_max_turn,
        final_max_turn=final_max_turn,
        rollouts_per_stage=rollouts_per_stage,
        initial_hint_prob=initial_hint_prob,
        final_hint_prob=final_hint_prob,
        warmup_rollouts=warmup_rollouts,
        hint_decay_mode=defaults.hint_decay_mode,
        hint_decay_optimizer_steps=defaults.hint_decay_optimizer_steps,
        mcts_warmup_optimizer_steps=mcts_warmup_optimizer_steps,
        initial_mcts_sims=defaults.initial_mcts_sims,
        final_mcts_sims=defaults.final_mcts_sims,
    )

    trace_logger: EpisodeTraceLogger | None = None
    trace_enabled = is_truthy_env(os.environ.get("EPISODE_TRACE_ENABLED", "1"))
    trace_dir = os.environ.get("EPISODE_TRACE_DIR", "").strip()
    if trace_enabled and trace_dir:
        try:
            trace_logger = EpisodeTraceLogger(
                trace_dir=trace_dir, rank=rank, game_name=game_name
            )
        except Exception as e:
            print(f"[EPISODE_TRACE] Failed to initialize logger: {e}")
    elif rank == 0:
        print(
            "[EPISODE_TRACE] Disabled "
            "(set EPISODE_TRACE_ENABLED=1 and EPISODE_TRACE_DIR)"
        )

    state = {
        "initialized": True,
        "rank": rank,
        "env_pool": env_pool,
        "num_servers": len(env_pool),
        "thread_pool": ThreadPoolExecutor(max_workers=len(env_pool)),
        "generation_semaphore": Semaphore(1),
        "curriculum": curriculum,
        "trace_logger": trace_logger,
    }
    _ROLLOUT_STATES[game_name] = state

    print(
        f"[CURRICULUM:{game_name}] Initialized: "
        f"initial_max_turn={initial_max_turn}, final_max_turn={final_max_turn}, "
        f"rollouts_per_stage={rollouts_per_stage}, warmup_rollouts={warmup_rollouts}, "
        f"hint=({initial_hint_prob}->{final_hint_prob}, mode={defaults.hint_decay_mode})"
    )
    return state


# ---------------------------------------------------------------------------
# Fallback result dicts
# ---------------------------------------------------------------------------

def _last_prompt_fallback() -> dict:
    return {
        "prompt_ids": [1],
        "completion_ids": [1],
        "logprobs": [1.0],
        "reward": 0.0,
        "final_score": 0.0,
    }


def _full_prompt_fallback() -> dict:
    return {
        "prompt_ids": [1],
        "completion_ids": [1],
        "action_mask": [0],
        "logprobs": [1.0],
        "reward": 0.0,
        "final_score": 0.0,
    }


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def rollout_parallelized_curriculum(
    game: GameEnvironment,
    prompts: list[str],
    trainer,
    include_action_mask: bool,
) -> dict[str, list]:
    """Shared rollout loop. All games funnel through this function."""
    state = _ensure_initialized(game, trainer)
    config = game.get_config()
    mode = "full" if include_action_mask else "last"

    # Lazy trl import: keeps the package importable in tooling environments
    # that don't have trl installed (e.g. static analysis, unit tests).
    from trl.experimental.openenv import generate_rollout_completions

    curriculum: CurriculumScheduler = state["curriculum"]
    trace_logger: EpisodeTraceLogger | None = state["trace_logger"]
    rank: int = state["rank"]
    env_pool: list[dict] = state["env_pool"]
    num_servers: int = state["num_servers"]
    generation_semaphore: Semaphore = state["generation_semaphore"]

    tokenizer = trainer.processing_class
    timeout = config.request_timeout_seconds
    optimizer_step = getattr(getattr(trainer, "state", None), "global_step", 0)

    current_max_turn = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob(optimizer_step)
    current_mcts_sims = curriculum.get_mcts_sims(optimizer_step)

    # Update MCTS config if the curriculum dictates a different sim count
    effective_mcts_config = dict(config.mcts_config)
    if current_mcts_sims is not None:
        effective_mcts_config["mcts_max_simulations"] = current_mcts_sims

    print(
        f"[CURRICULUM:{config.game_name}] rollouts={curriculum.total_rollouts} "
        f"step={optimizer_step}: max_turn={current_max_turn} "
        f"hint_prob={current_hint_prob:.2f} mcts_sims={current_mcts_sims}"
    )

    def run_single_prompt(index: int, prompt: str) -> tuple[int, dict | None]:
        game_id = int(prompt)
        server = env_pool[(index + rank) % num_servers]
        env_endpoint = server["base_url"]

        invalid_count = 0
        done = False
        final_reward = 0.0
        turn_number = 0
        accumulated_shaping_reward = 0.0
        step_rewards: list[float] = []
        step_records: list[dict] = []
        termination_reason = "unknown"

        # Token bookkeeping (identical across games)
        if include_action_mask:
            episode_prompt_ids: list[int] = []
            episode_completion_ids: list[int] = []
            episode_logprobs: list[float] = []
            episode_action_mask: list[int] = []
            prev_full_ids: list[int] | None = None
        else:
            prompt_ids_last: list[int] = []
            completion_ids_last: list[int] = []
            logprobs_last: list[float] = []

        # --- Reset environment ---
        seed = random.randint(0, 2**31 - 1)
        reset_payload = game.build_reset_payload(
            game_id=game_id,
            mcts_config=effective_mcts_config,
            seed=seed,
        )
        try:
            episode_id, raw_observation = reset_environment(
                env_endpoint=env_endpoint,
                payload=reset_payload,
                timeout=timeout,
            )
            observation = game.extract_and_format_observation(raw_observation)
        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            if trace_logger and trace_logger.should_log():
                trace_logger.log_episode(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "game_id": game_id,
                        "environment": config.game_name,
                        "status": "reset_failed",
                        "error": str(e),
                    }
                )
            return index, None

        episode_state = game.on_episode_start(observation)
        use_hints = random.random() < current_hint_prob
        messages = [
            {"role": "system", "content": game.get_system_prompt(use_hints)},
            {"role": "user", "content": observation},
        ]

        # --- Interaction loop ---
        while not done and turn_number < current_max_turn:
            observation_before = observation
            legal_action_map = game.extract_legal_action_map(observation_before)
            state_features = game.extract_state_features(observation_before)

            # Strategy forcing: let the game inject an action without model calls
            forced_action = game.pre_model_strategy_override(
                messages=messages,
                observation=observation_before,
                turn_number=turn_number,
                max_turn=current_max_turn,
                episode_state=episode_state,
                mode=mode,
            )

            if forced_action is not None:
                # Skip model generation entirely
                action_id = forced_action
                action_label = legal_action_map.get(action_id, "")
                parse_failed = False
                completion_text = ""
                # Record the forced action in the conversation
                messages.append({"role": "assistant", "content": action_id})
            else:
                if not legal_action_map:
                    accumulated_shaping_reward -= config.invalid_action_penalty
                    termination_reason = "no_legal_actions"
                    break

                # --- Generate model completion ---
                with generation_semaphore:
                    rollout_outputs = generate_rollout_completions(
                        trainer, prompts=[messages], as_chat=True
                    )[0]

                prompt_ids = rollout_outputs.get("prompt_ids", [])
                completion_ids = rollout_outputs.get("completion_ids", [])
                logprobs = rollout_outputs.get("logprobs", [])
                completion_text = tokenizer.decode(
                    completion_ids, skip_special_tokens=True
                ).strip()

                # --- Token bookkeeping ---
                if include_action_mask:
                    if len(prompt_ids) > config.max_prompt_len:
                        print(
                            f"Warning: Prompt exceeded {config.max_prompt_len} tokens "
                            f"({len(prompt_ids)}) at turn {turn_number}"
                        )
                        termination_reason = "max_prompt_len_exceeded"
                        break

                    if turn_number == 0:
                        episode_prompt_ids = prompt_ids
                        prev_full_ids = prompt_ids.copy()
                    else:
                        if prev_full_ids is None:
                            prev_full_ids = prompt_ids.copy()
                        elif prompt_ids[: len(prev_full_ids)] != prev_full_ids:
                            prev_full_ids = prompt_ids.copy()
                        else:
                            delta_prompt_ids = prompt_ids[len(prev_full_ids):]
                            if delta_prompt_ids:
                                episode_completion_ids.extend(delta_prompt_ids)
                                episode_logprobs.extend([0.0] * len(delta_prompt_ids))
                                episode_action_mask.extend([0] * len(delta_prompt_ids))
                            prev_full_ids = prompt_ids.copy()

                    if completion_ids:
                        episode_completion_ids.extend(completion_ids)
                        episode_logprobs.extend(logprobs)
                        episode_action_mask.extend([1] * len(completion_ids))
                        if prev_full_ids is not None:
                            prev_full_ids = prev_full_ids + completion_ids
                else:
                    prompt_ids_last = prompt_ids
                    completion_ids_last = completion_ids
                    logprobs_last = logprobs

                messages.append({"role": "assistant", "content": completion_text})

                # --- Parse action ---
                action_id = game.parse_action_id(completion_text, legal_action_map)
                parse_failed = not action_id
                if parse_failed or action_id not in legal_action_map:
                    invalid_count += 1
                    accumulated_shaping_reward -= config.invalid_action_penalty
                    action_id = game.select_fallback_action(
                        legal_action_map, state_features
                    )

                action_label = legal_action_map.get(action_id, "")

            # --- Reward shaping ---
            shaping = game.compute_step_shaping_reward(
                state_features=state_features,
                action_id=action_id,
                action_label=action_label,
                legal_action_map=legal_action_map,
                episode_state=episode_state,
                mode=mode,
            )
            accumulated_shaping_reward += shaping

            # --- Step environment ---
            try:
                raw_observation, step_reward, done, step_block = step_environment(
                    env_endpoint=env_endpoint,
                    episode_id=episode_id,
                    action_to_send=action_id,
                    timeout=timeout,
                )
                observation = game.extract_and_format_observation(raw_observation)
            except Exception as e:
                print(f"Step failed: {e}")
                observation = ""
                step_reward = -0.01
                done = False
                invalid_count += 1
                accumulated_shaping_reward -= config.invalid_action_penalty
                step_block = {"reward": step_reward, "done": False}

            obs_lower = observation.lower()
            invalid_or_noop = (
                "invalid" in obs_lower
                or "nothing happens" in obs_lower
                or "nothing happened" in obs_lower
            )
            if invalid_or_noop:
                invalid_count += 1
                accumulated_shaping_reward -= config.invalid_action_penalty

            episode_state = game.on_step_complete(
                observation=observation,
                step_reward=step_reward,
                done=done,
                is_invalid=invalid_or_noop,
                action_id=action_id,
                episode_state=episode_state,
            )

            step_rewards.append(step_reward)

            if done:
                final_reward = game.extract_terminal_reward(step_block, observation)
                termination_reason = "done"
            else:
                messages.append({"role": "user", "content": observation})

            if trace_logger is not None:
                step_records.append(
                    game.build_step_trace_record(
                        turn=turn_number,
                        completion_text=completion_text,
                        action_id=action_id,
                        action_label=action_label,
                        state_features=state_features,
                        shaping_reward=shaping,
                        observation_before=observation_before,
                        observation_after=observation,
                        step_reward=step_reward,
                        done=done,
                        invalid_or_noop=invalid_or_noop,
                        parse_failed=parse_failed,
                        trace_logger=trace_logger,
                    )
                )

            turn_number += 1

        # --- Episode finalization ---
        if not done:
            if termination_reason == "unknown":
                termination_reason = "max_turn_reached"

        train_reward = game.compute_episode_reward(
            final_reward=final_reward if done else 0.0,
            accumulated_shaping=accumulated_shaping_reward,
            done=done,
            episode_state=episode_state,
            step_rewards=step_rewards,
            mode=mode,
        )

        clipped_shaping = clamp(
            accumulated_shaping_reward,
            -config.shaping_reward_clip,
            config.shaping_reward_clip,
        )

        print(
            f"[{config.game_name} ID:{game_id} Done:{int(done)} T:{turn_number:2d} "
            f"Env:{final_reward:+.3f} Shape:{accumulated_shaping_reward:+.3f} "
            f"ClipShape:{clipped_shaping:+.3f} Train:{train_reward:+.3f} "
            f"Inv:{invalid_count}]"
        )

        if trace_logger and trace_logger.should_log():
            trace_logger.log_episode(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "game_id": game_id,
                    "episode_id": episode_id,
                    "environment": config.game_name,
                    "status": "completed" if done else "truncated",
                    "termination_reason": termination_reason,
                    "turns": turn_number,
                    "final_reward": float(final_reward),
                    "raw_shaping_reward": float(accumulated_shaping_reward),
                    "clipped_shaping_reward": float(clipped_shaping),
                    "train_reward": float(train_reward),
                    "invalid_count": invalid_count,
                    "steps": step_records,
                }
            )

        if include_action_mask:
            if len(episode_completion_ids) > config.max_episode_tokens:
                episode_completion_ids = episode_completion_ids[: config.max_episode_tokens]
                episode_logprobs = episode_logprobs[: config.max_episode_tokens]
                episode_action_mask = episode_action_mask[: config.max_episode_tokens]

            return index, {
                "prompt_ids": episode_prompt_ids,
                "completion_ids": episode_completion_ids,
                "action_mask": episode_action_mask,
                "logprobs": episode_logprobs,
                "reward": train_reward,
                "final_score": final_reward,
            }

        return index, {
            "prompt_ids": prompt_ids_last,
            "completion_ids": completion_ids_last,
            "logprobs": logprobs_last,
            "reward": train_reward,
            "final_score": final_reward,
        }

    # --- Parallel execution ---
    executor: ThreadPoolExecutor = state["thread_pool"]
    fallback_builder = _full_prompt_fallback if include_action_mask else _last_prompt_fallback

    results: list[dict | None] = [None] * len(prompts)
    futures = [executor.submit(run_single_prompt, i, p) for i, p in enumerate(prompts)]
    for future in as_completed(futures):
        idx, res = future.result()
        results[idx] = res if res is not None else fallback_builder()

    list_results = [r for r in results if r is not None]
    curriculum.step(len(prompts))

    # --- Batch statistics ---
    finished = sum(1 for r in list_results if r["final_score"] != 0)
    avg_return = (
        sum(r["reward"] for r in list_results) / len(list_results)
        if list_results
        else 0.0
    )
    print(
        f"[BATCH:{config.game_name}] Finished: {finished}/{len(list_results)}, "
        f"AvgReturn: {avg_return:.3f}"
    )

    if include_action_mask:
        return {
            "prompt_ids": [r["prompt_ids"] for r in list_results],
            "completion_ids": [r["completion_ids"] for r in list_results],
            "action_mask": [r["action_mask"] for r in list_results],
            "logprobs": [r["logprobs"] for r in list_results],
            "env_rewards": [r["reward"] for r in list_results],
        }

    return {
        "prompt_ids": [r["prompt_ids"] for r in list_results],
        "completion_ids": [r["completion_ids"] for r in list_results],
        "logprobs": [r["logprobs"] for r in list_results],
        "env_rewards": [r["reward"] for r in list_results],
    }


# ---------------------------------------------------------------------------
# Public API builders (factory returns the pair of rollout functions + reward)
# ---------------------------------------------------------------------------

def make_rollout_functions(game: GameEnvironment):
    """Return (rollout_full, rollout_last, reward_func) bound to this game."""

    def rollout_full_prompt_and_completion_parallelized_curriculum(
        prompts: list[str],
        trainer,
        max_turns: int = 30,
    ) -> dict[str, list]:
        del max_turns  # curriculum controls effective horizon
        return rollout_parallelized_curriculum(
            game=game, prompts=prompts, trainer=trainer, include_action_mask=True
        )

    def rollout_last_prompt_and_completion_parallelized_curriculum(
        prompts: list[str],
        trainer,
        max_turns: int = 30,
    ) -> dict[str, list]:
        del max_turns
        return rollout_parallelized_curriculum(
            game=game, prompts=prompts, trainer=trainer, include_action_mask=False
        )

    def rollout_reward_func(completions, **kwargs):
        rewards = kwargs.get("env_rewards") if kwargs else None
        return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)

    return (
        rollout_full_prompt_and_completion_parallelized_curriculum,
        rollout_last_prompt_and_completion_parallelized_curriculum,
        rollout_reward_func,
    )
