"""Leduc Poker game environment.

Game-specific pieces only: rules text, card parsing, hand-strength scoring,
action classification into fold/check/call/raise, and pot-aware shaping.
"""

from __future__ import annotations

import re

from game_envs.base import CurriculumDefaults, GameConfig, GameEnvironment
from game_envs.shared import clamp, remove_reasoning_tags


# ---------------------------------------------------------------------------
# Reward-shaping constants
# ---------------------------------------------------------------------------

INVALID_ACTION_PENALTY = 0.10
FOLD_WITH_STRONG_HAND_PENALTY = 0.05
VALUE_BET_BONUS = 0.05
BLUFF_QUALITY_BONUS = 0.0
PASSIVE_WITH_STRONG_PENALTY = 0.0
FOLD_WEAK_HAND_BONUS = 0.0
SHAPING_REWARD_CLIP = 0.20
TERMINAL_REWARD_CLIP = 1.00

CARD_RANK = {"J": 0, "Q": 1, "K": 2}

STRATEGY_TIPS = """
STRATEGY TIPS:
- With a King: Raise for value, especially if it pairs the community card.
- With a Queen: Play cautiously. Call most bets; raise only with a pair.
- With a Jack: Minimize losses. Check/call cheaply; fold against heavy aggression unless you have a pair.
- PAIR (your card matches community card): Always raise aggressively.
- NO PAIR post-flop with low card: Prefer checking/calling. Fold vs large raises.
- Do NOT raise every hand. Selective aggression wins; constant raising is exploitable.
"""

SYSTEM_PROMPT_BASE = """You are playing leduc_poker.

# Game Rules
LEDUC POKER RULES:

Deck: 2 suits x 3 ranks. For 2 players: 6 cards (J, Q, K in two suits).

Setup: Each player starts with 100 chips, pays 1 ante. Two rounds of betting.

Round 1: Each player receives one private card.
Actions: Fold (lose ante), Call/Check (match current bet or pass), Raise (add 2 chips to bet).
Maximum 2 raises per round.

Round 2: One public card is revealed. Same actions, but Raise adds 4 chips.

Winning: Player with best hand wins pot (or last remaining if others fold).
Hand ranking (high to low): Pair (private + public match) > High card value (K > Q > J).

# Output Format
You must respond with ONLY the action ID (a single number).
Do NOT include descriptions or explanations.
Examples:
- For action "0 -> Fold": respond "0"
- For action "1 -> Call": respond "1"
- For action "2 -> Raise": respond "2"

CRITICAL: Your entire response must be a single number. No words, no punctuation, no explanation.
"""


def _hand_strength(hole_card: str | None, community_card: str | None) -> float:
    """[0, 1] hand-strength estimate. See original leduc file for ranking."""
    if hole_card is None:
        return 0.5
    rank_val = CARD_RANK.get(hole_card, 0)
    if community_card is None:
        return rank_val / 2.0
    if hole_card == community_card:
        return 0.8 + 0.1 * rank_val  # 0.8, 0.9, 1.0 for J/Q/K pair
    return rank_val / 4.0


def _classify_action(label: str) -> str:
    low = (label or "").strip().lower()
    if "fold" in low:
        return "fold"
    if "call" in low or "check" in low:
        return "call"
    if "raise" in low or "bet" in low:
        return "raise"
    return "unknown"


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class LeducPokerEnvironment(GameEnvironment):
    def get_config(self) -> GameConfig:
        return GameConfig(
            game_name="leduc_poker",
            selected_game="leduc_poker",
            mcts_config={
                "opponent": "mcts",
                "mcts_max_simulations": 50,
                "mcts_num_rollouts": 1,
            },
            invalid_action_penalty=INVALID_ACTION_PENALTY,
            shaping_reward_clip=SHAPING_REWARD_CLIP,
            terminal_reward_clip=TERMINAL_REWARD_CLIP,
        )

    def get_curriculum_defaults(self) -> CurriculumDefaults:
        return CurriculumDefaults(
            initial_max_turn=4,
            final_max_turn=8,
            rollouts_per_stage=1280,
            initial_hint_prob=0.5,
            final_hint_prob=0.0,
            warmup_rollouts=128,
            hint_decay_mode="rollout",
        )

    def get_system_prompt(self, use_hints: bool) -> str:
        prompt = SYSTEM_PROMPT_BASE
        if use_hints:
            prompt += "\n" + STRATEGY_TIPS
        return prompt

    def extract_and_format_observation(self, raw_observation: str) -> str:
        return raw_observation or ""

    def extract_state_features(self, observation: str) -> dict:
        hole_card: str | None = None
        community_card: str | None = None
        pot_size: int = 0
        round_number: int = 1

        hole_match = re.search(
            r"(?:Your card|Your hole card|Hand|Private card):\s*\[?([JQK])\]?",
            observation,
            flags=re.IGNORECASE,
        )
        if hole_match:
            hole_card = hole_match.group(1).upper()

        comm_match = re.search(
            r"(?:Community card|Public card|Board|Flop):\s*\[?([JQK])\]?",
            observation,
            flags=re.IGNORECASE,
        )
        if comm_match:
            community_card = comm_match.group(1).upper()

        pot_match = re.search(r"(?:Pot|Pot size):\s*(\d+)", observation, flags=re.IGNORECASE)
        if pot_match:
            pot_size = int(pot_match.group(1))

        round_match = re.search(
            r"(?:Round|Betting round):\s*(\d+)", observation, flags=re.IGNORECASE
        )
        if round_match:
            round_number = int(round_match.group(1))
        elif community_card is not None:
            round_number = 2

        return {
            "hole_card": hole_card,
            "community_card": community_card,
            "pot_size": pot_size,
            "round_number": round_number,
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

        normalized = cleaned.strip().lower()
        for action_id, label in legal_action_map.items():
            if normalized == label.strip().lower():
                return action_id

        for keyword in ("fold", "check", "call", "raise", "bet"):
            if keyword in normalized:
                for action_id, label in legal_action_map.items():
                    if keyword in label.lower():
                        return action_id

        return ""

    def select_fallback_action(
        self,
        legal_action_map: dict[str, str],
        state_features: dict,
    ) -> str:
        # Prefer check > call > lowest action id
        for action_id, label in legal_action_map.items():
            if _classify_action(label) == "check":
                return action_id
        for action_id, label in legal_action_map.items():
            if _classify_action(label) == "call":
                return action_id
        return sorted(legal_action_map.keys(), key=lambda x: int(x))[0]

    def compute_step_shaping_reward(
        self,
        state_features: dict,
        action_id: str,
        action_label: str,
        legal_action_map: dict[str, str],
        episode_state=None,
        mode: str = "full",
    ) -> float:
        hole_card = state_features.get("hole_card")
        community_card = state_features.get("community_card")
        round_number = state_features.get("round_number", 1)
        strength = _hand_strength(hole_card, community_card)
        action_type = _classify_action(action_label)

        reward = 0.0
        can_raise = any(
            _classify_action(lbl) == "raise" for lbl in legal_action_map.values()
        )

        if action_type == "fold" and strength >= 0.6:
            reward -= FOLD_WITH_STRONG_HAND_PENALTY
        if action_type == "raise" and strength >= 0.7:
            reward += VALUE_BET_BONUS
        if action_type == "raise" and round_number == 1 and strength <= 0.5:
            reward += BLUFF_QUALITY_BONUS
        if (
            action_type == "call"
            and round_number == 2
            and strength >= 0.7
            and can_raise
        ):
            reward -= PASSIVE_WITH_STRONG_PENALTY
        if action_type == "fold" and round_number == 2 and strength < 0.3:
            reward += FOLD_WEAK_HAND_BONUS

        return reward

    def extract_terminal_reward(self, step_block: dict, observation_text: str) -> float:
        step_reward = step_block.get("reward", 0.0) if isinstance(step_block, dict) else 0.0
        try:
            value = float(step_reward)
        except Exception:
            value = 0.0
        return clamp(value, -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)

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
        record["action_type"] = _classify_action(action_label)
        record["hand_strength"] = _hand_strength(
            state_features.get("hole_card"),
            state_features.get("community_card"),
        )
        return record
