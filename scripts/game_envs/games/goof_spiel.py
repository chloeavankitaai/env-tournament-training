"""Goofspiel game environment.

Two unusual pieces this game needs:

1. **Legal actions**: the server observation for goofspiel does not include a
   ``Legal Actions:`` block — we reconstruct it from the player's hand.
2. **Strategy forcing** (``last`` mode only): for all turns except the final
   one, we play the expert move (bid the card whose value matches the prize
   card) rather than letting the model generate. The model only sees and
   makes the *last* turn, which is what we train on. Used to bootstrap a
   learning signal before the model has learned the game.

In ``full`` mode, strategy forcing is disabled; the model plays every turn
and we use per-step shaping (``STEP_STRATEGY_REWARD``) to reward strategy
adherence while ``all_steps_correct`` remains true.
"""

from __future__ import annotations

import re
from typing import Any

from game_envs.base import CurriculumDefaults, GameConfig, GameEnvironment
from game_envs.shared import clamp, remove_reasoning_tags


# ---------------------------------------------------------------------------
# Reward-shaping constants
# ---------------------------------------------------------------------------

INVALID_ACTION_PENALTY = 0.05
STEP_STRATEGY_REWARD = 0.10
STRATEGY_REWARD_WEIGHT = 0.50   # weight of strategy ratio in ``full`` mode terminal blend
LAST_STRATEGY_REWARD = 1.00     # ``last`` mode base reward for strategy adherence
LAST_INVALID_PENALTY = 0.10     # ``last`` mode invalid-action penalty
SHAPING_REWARD_CLIP = 1.50      # goof_spiel accumulates many per-step bonuses
TERMINAL_REWARD_CLIP = 3.00


# ---------------------------------------------------------------------------
# Observation parsing helpers
# ---------------------------------------------------------------------------

def _extract_player_id(obs_text: str) -> int:
    match = re.search(r"You are Player (\d+)", obs_text)
    return int(match.group(1)) if match else 0


def _extract_hand_cards(obs_text: str, player_id: int = 0) -> list[int]:
    pattern = rf"P{player_id} hand:\s*([\d ]+)"
    match = re.search(pattern, obs_text)
    if not match:
        return []
    return [int(card) for card in match.group(1).strip().split()]


def _extract_prize_card(obs_text: str) -> int | None:
    match = re.search(r"Current point card:\s*(\d+)", obs_text)
    return int(match.group(1)) if match else None


def _format_goof_spiel_observation(obs_text: str) -> str:
    """Reformat the server observation to include a ``Legal Actions`` block."""
    if not obs_text:
        return ""
    if "Invalid action:" in obs_text and "Legal Actions:" in obs_text:
        return obs_text

    state_match = re.search(r"Current State:\n(.*)", obs_text, re.DOTALL)
    if not state_match:
        return obs_text
    state_text = state_match.group(0)

    # Drop "Waiting for Player -2 to move..." if present
    state_text = re.sub(
        r"\n\nWaiting for Player -2 to move\.\.\.$", "", state_text
    )

    player_id = _extract_player_id(obs_text)
    hand_match = re.search(rf"P{player_id} hand: ([\d\s]+)", state_text)
    if not hand_match:
        return state_text

    cards = [int(card) for card in hand_match.group(1).strip().split()]
    # The action id is the card value - 1 (cards are 1..N).
    legal_actions = [
        f"{card - 1} -> [P{player_id}]Bid: {card}" for card in cards
    ]
    formatted = (
        state_text
        + "\n\nYou are Player "
        + str(player_id)
        + ".\nLegal Actions:\n"
        + "\n".join(legal_actions)
        + "\n\nYour choice (ID only):"
    )
    return formatted


def _bid_card_from_action_id(action_id: str) -> int | None:
    try:
        return int(action_id.strip()) + 1
    except Exception:
        return None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = (
    "You are playing goofspiel.\n\n"
    "# Game Rules\n"
    "GOOFSPIEL RULES:\n"
    "Setup: Each player has bid cards numbered 1 to N. A prize deck with "
    "cards 1 to N is shuffled.\n"
    "Goal: Win the most points by bidding on prize cards.\n\n"
    "Each turn:\n"
    "1. Reveal top prize card (worth its face value in points)\n"
    "2. Players simultaneously play one bid card from their hand\n"
    "3. Highest bidder wins the prize card (adds its value to score)\n"
    "4. If bids tie, prize card is discarded (no one gets points)\n\n"
    "Winning: Player with most points after all rounds wins.\n\n\n"
    "# Output Format\n"
    "You must respond with ONLY the action ID (a single number).\n"
    "Do NOT include descriptions or explanations.\n\n"
    "Examples:\n"
    '- For action "0 -> roll": respond "0"\n'
    '- For action "89 -> a3": respond "89"'
)

STRATEGY_TIPS = (
    "\n\nThe best strategies is to bid the card with same value as the point card \n\n"
    "Example: \n"
    "If the point card is 1, bid using card 1, likely action ID 0\n"
    "If the point card is 13, bid using card 13, likely action ID 12\n"
    "If the point card is 10, bid using card 10, likely action ID 9\n"
    "Always bid following this strategy to maximize your winning chance."
)


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class GoofSpielEnvironment(GameEnvironment):
    def get_config(self) -> GameConfig:
        return GameConfig(
            game_name="goof_spiel",
            selected_game="goofspiel",
            mcts_config={"opponent": "mcts"},
            invalid_action_penalty=INVALID_ACTION_PENALTY,
            shaping_reward_clip=SHAPING_REWARD_CLIP,
            terminal_reward_clip=TERMINAL_REWARD_CLIP,
        )

    def get_curriculum_defaults(self) -> CurriculumDefaults:
        return CurriculumDefaults(
            initial_max_turn=1,
            final_max_turn=13,
            rollouts_per_stage=1280,
            initial_hint_prob=0.75,
            final_hint_prob=0.0,
            warmup_rollouts=1280,
            hint_decay_mode="rollout",
        )

    def get_system_prompt(self, use_hints: bool) -> str:
        prompt = SYSTEM_PROMPT_BASE
        if use_hints:
            prompt += STRATEGY_TIPS
        return prompt

    def extract_and_format_observation(self, raw_observation: str) -> str:
        return _format_goof_spiel_observation(raw_observation or "")

    def extract_state_features(self, observation: str) -> dict:
        player_id = _extract_player_id(observation)
        return {
            "player_id": player_id,
            "prize_card": _extract_prize_card(observation),
            "hand_cards": _extract_hand_cards(observation, player_id=player_id),
        }

    def parse_action_id(
        self,
        completion_text: str,
        legal_action_map: dict[str, str],
    ) -> str:
        if not legal_action_map:
            return ""
        cleaned = remove_reasoning_tags(completion_text or "")
        cleaned = cleaned.removesuffix("</s>")
        if "Action:" in cleaned:
            cleaned = cleaned.split("Action:")[-1].strip()

        for num in re.findall(r"-?\d+", cleaned):
            if num in legal_action_map:
                return num

        # Fallback: strict int parse (original goof_spiel accepted raw ints
        # without validating against legal_action_map).
        try:
            raw = int(cleaned.strip())
            return str(raw)
        except Exception:
            return ""

    def select_fallback_action(
        self,
        legal_action_map: dict[str, str],
        state_features: dict,
    ) -> str:
        # Expert move: bid the card matching the prize card.
        prize_card = state_features.get("prize_card")
        if prize_card is not None:
            candidate = str(prize_card - 1)
            if candidate in legal_action_map:
                return candidate
        if legal_action_map:
            return sorted(legal_action_map.keys(), key=lambda x: int(x))[0]
        return ""

    # ---- Episode-state hooks ----

    def on_episode_start(self, observation: str) -> Any:
        return {
            "total_strategy_opportunities": 0,
            "strategy_followed_count": 0,
            "all_steps_correct": True,
            "invalid_count": 0,
            "last_bid_followed_strategy": False,
            "last_bid_was_invalid": False,
        }

    def on_step_complete(
        self,
        observation: str,
        step_reward: float,
        done: bool,
        is_invalid: bool,
        action_id: str,
        episode_state: Any,
    ) -> Any:
        if not isinstance(episode_state, dict):
            episode_state = self.on_episode_start("")
        if is_invalid:
            episode_state["invalid_count"] = episode_state.get("invalid_count", 0) + 1
            episode_state["all_steps_correct"] = False
        return episode_state

    def compute_step_shaping_reward(
        self,
        state_features: dict,
        action_id: str,
        action_label: str,
        legal_action_map: dict[str, str],
        episode_state=None,
        mode: str = "full",
    ) -> float:
        """Per-step strategy adherence bonus (only in ``full`` mode).

        Also updates ``episode_state`` counters in place so that
        ``compute_episode_reward`` can use them. ``last`` mode does its
        shaping during ``compute_episode_reward`` based on the final turn.
        """
        if not isinstance(episode_state, dict):
            return 0.0

        prize_card = state_features.get("prize_card")
        bid_card = _bid_card_from_action_id(action_id)
        is_strategy = bid_card is not None and bid_card == prize_card

        if mode == "last":
            # Only the final (model-generated) turn matters for last mode,
            # which is handled in compute_episode_reward.
            episode_state["last_bid_followed_strategy"] = bool(is_strategy)
            episode_state["last_bid_was_invalid"] = bid_card is None
            return 0.0

        # ``full`` mode: immediate reward per step while still perfect.
        episode_state["total_strategy_opportunities"] = (
            episode_state.get("total_strategy_opportunities", 0) + 1
        )
        if bid_card is None:
            episode_state["all_steps_correct"] = False
            return 0.0

        if is_strategy and episode_state.get("all_steps_correct", True):
            episode_state["strategy_followed_count"] = (
                episode_state.get("strategy_followed_count", 0) + 1
            )
            return STEP_STRATEGY_REWARD

        episode_state["all_steps_correct"] = False
        return 0.0

    def pre_model_strategy_override(
        self,
        messages: list[dict],
        observation: str,
        turn_number: int,
        max_turn: int,
        episode_state: Any,
        mode: str = "full",
    ) -> str | None:
        # Only the ``last`` variant forces expert moves before the final turn.
        if mode != "last":
            return None
        if turn_number >= max_turn - 1:
            return None
        hand_cards = _extract_hand_cards(observation)
        if len(hand_cards) <= 1:
            return None
        prize_card = _extract_prize_card(observation)
        if prize_card is None:
            return None
        return str(prize_card - 1)

    def extract_terminal_reward(self, step_block: dict, observation_text: str) -> float:
        step_reward = step_block.get("reward", 0.0) if isinstance(step_block, dict) else 0.0
        try:
            return float(step_reward)
        except Exception:
            return 0.0

    def compute_episode_reward(
        self,
        final_reward: float,
        accumulated_shaping: float,
        done: bool,
        episode_state: Any,
        step_rewards: list[float],
        mode: str = "full",
    ) -> float:
        state = episode_state if isinstance(episode_state, dict) else {}
        invalid_count = int(state.get("invalid_count", 0))

        if mode == "last":
            # Reward is driven by whether the single model-generated turn
            # followed the expert strategy.
            if state.get("last_bid_was_invalid", False):
                return -LAST_INVALID_PENALTY
            if state.get("last_bid_followed_strategy", False):
                return LAST_STRATEGY_REWARD
            return 0.0

        # ``full`` mode: blend strategy-ratio + env reward, plus the
        # accumulated per-step shaping the engine already tracks.
        total_opp = int(state.get("total_strategy_opportunities", 0))
        followed = int(state.get("strategy_followed_count", 0))
        strategy_ratio = (followed / total_opp) if total_opp > 0 else 0.0

        immediate_rewards = float(accumulated_shaping)
        if not done:
            shaped = immediate_rewards + strategy_ratio
        else:
            shaped = (
                STRATEGY_REWARD_WEIGHT * strategy_ratio
                + (1.0 - STRATEGY_REWARD_WEIGHT) * float(final_reward)
                + immediate_rewards
            )
        shaped -= 0.05 * float(invalid_count)
        return clamp(shaped, -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)

    def build_step_trace_record(
        self,
        *,
        turn,
        completion_text,
        action_id,
        action_label,
        state_features,
        shaping_reward,
        observation_before,
        observation_after,
        step_reward,
        done,
        invalid_or_noop,
        parse_failed,
        trace_logger,
    ) -> dict:
        record = super().build_step_trace_record(
            turn=turn,
            completion_text=completion_text,
            action_id=action_id,
            action_label=action_label,
            state_features=state_features,
            shaping_reward=shaping_reward,
            observation_before=observation_before,
            observation_after=observation_after,
            step_reward=step_reward,
            done=done,
            invalid_or_noop=invalid_or_noop,
            parse_failed=parse_failed,
            trace_logger=trace_logger,
        )
        record["prize_card"] = state_features.get("prize_card")
        bid_card = _bid_card_from_action_id(action_id)
        record["bid_card"] = bid_card
        record["strategy_followed"] = (
            bid_card is not None and bid_card == state_features.get("prize_card")
        )
        return record
