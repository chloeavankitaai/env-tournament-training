"""HTTP client helpers for talking to the environment server.

These were previously duplicated across liars_dice/leduc_poker as
``_build_env_pool``, ``_reset_environment``, and ``_step_environment``.
"""

from __future__ import annotations

from typing import Any

import requests

from game_envs.shared import safe_float


def build_env_pool(
    server_urls: list[str],
    init_task_id: int,
    mcts_config: dict[str, Any],
    init_timeout_seconds: int,
) -> list[dict[str, str]]:
    """Perform a one-time health-check reset against each server.

    Returns a list of dicts with at least a ``base_url`` key, in the order
    the caller provided. Raises ``RuntimeError`` on the first failure — we
    want training to fail loudly if an env server is unreachable.
    """
    env_pool: list[dict[str, str]] = []

    for idx, base_url in enumerate(server_urls):
        try:
            print(f"[INIT] Initializing env on server {idx}: {base_url}")
            # Static seed is fine here — this is a one-time health check.
            payload = {"task_id": init_task_id, "seed": 42, **mcts_config}
            res = requests.post(
                f"{base_url}/reset",
                json=payload,
                timeout=init_timeout_seconds,
            )
            res.raise_for_status()
            env_pool.append({"base_url": base_url})
            print(f"[INIT] Server {idx} ready")
        except Exception as e:
            raise RuntimeError(f"Failed to init server {base_url}: {e}") from e

    return env_pool


def reset_environment(
    env_endpoint: str,
    payload: dict,
    timeout: int,
) -> tuple[str, str]:
    """POST /reset. Returns ``(episode_id, raw_observation)``."""
    res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=timeout)
    res.raise_for_status()
    result_block = res.json()["result"]
    episode_id = result_block.get("episode_id", "")
    raw_observation = result_block.get("observation", "")
    return episode_id, raw_observation


def step_environment(
    env_endpoint: str,
    episode_id: str,
    action_to_send: str,
    timeout: int,
) -> tuple[str, float, bool, dict]:
    """POST /step. Returns ``(raw_observation, reward, done, step_block)``."""
    step_payload = {"action": action_to_send, "episode_id": episode_id}
    res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=timeout)
    res.raise_for_status()
    step_block = res.json()["result"]
    raw_observation = step_block.get("observation", "")
    step_reward = safe_float(step_block.get("reward", 0.0), default=0.0)
    done = bool(step_block.get("done", False))
    return raw_observation, step_reward, done, step_block
