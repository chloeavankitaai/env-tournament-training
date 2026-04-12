"""Liar's Dice game environment.

Game-specific pieces only: system prompt, observation/action parsing,
bid-plausibility and challenge shaping, terminal reward extraction with
multiple fallback sources.
"""

from __future__ import annotations

import math
import re

from game_envs.base import CurriculumDefaults, GameConfig, GameEnvironment
from game_envs.shared import clamp, remove_reasoning_tags, safe_float


# ---------------------------------------------------------------------------
# Reward-shaping constants
# ---------------------------------------------------------------------------

INVALID_ACTION_PENALTY = 0.10
PASS_MISSED_CHALLENGE_PENALTY = 0.04
BID_PLAUSIBILITY_BONUS = 0.02
BID_PLAUSIBILITY_PENALTY = 0.02
SHAPING_REWARD_CLIP = 0.25
TERMINAL_REWARD_CLIP = 1.00

STRATEGY_TIPS = """
STRATEGY TIPS:
- Keep bids minimally stronger than current bid when uncertain.
- Use your own dice + wild 6s to estimate plausible total counts.
- Prefer calling Liar when the required quantity is implausibly high.
- Avoid large overbids unless your private dice strongly support it.
"""

SYSTEM_PROMPT_BASE = """You are playing liars_dice.

# Game Rules
LIAR'S DICE RULES:

Setup: Each player has N dice (1-5 depending on variant). All players roll their dice secretly.

Goal: Make bids about total dice across ALL players, or call "Liar" on opponent's bid.

Actions:
- Bid (quantity, face): Claim there are at least 'quantity' dice showing 'face' among all dice.
- Call Liar: Challenge the previous bid.

Bidding rules: Each bid must be higher than the previous bid. "Higher" means:
  - Same face value but higher quantity (e.g., "2 fours" beats "1 four")
  - Same quantity but higher face value (e.g., "2 fives" beats "2 fours")

Wild dice: 6s are WILD and count as ANY face value.
- When counting dice for a bid, include 6s in the count
- Example: Bid "3 fours" means at least 3 dice showing EITHER 4 OR 6

Winning: If you call Liar and previous bid was false, opponent loses. If bid was true or exact, you lose.

# Output Format
You must respond with ONLY the action ID (a single number).
Do NOT include descriptions or explanations.
Examples:
- For action "59 -> 10-6": respond "59"
- For action "60 -> Liar": respond "60"
"""


# ---------------------------------------------------------------------------
# Bid / dice helpers
# ---------------------------------------------------------------------------

def _extract_bid_tuple(label_or_text: str) -> tuple[int, int] | None:
    if not label_or_text:
        return None
    match = re.search(r"(\d+)\s*-\s*(\d+)", label_or_text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _is_liar_label(label: str) -> bool:
    return "liar" in (label or "").strip().lower()


def _bid_rank(bid: tuple[int, int]) -> int:
    quantity, face = bid
    return quantity * 6 + face


def _count_face_support(own_dice: list[int], target_face: int, wild_six_enabled: bool) -> int:
    if wild_six_enabled and target_face != 6:
        return sum(1 for value in own_dice if value == target_face or value == 6)
    return sum(1 for value in own_dice if value == target_face)


def _binomial_tail_probability(num_trials: int, success_prob: float, min_successes: int) -> float:
    if min_successes <= 0:
        return 1.0
    if num_trials <= 0:
        return 0.0
    success_prob = clamp(success_prob, 0.0, 1.0)
    tail = 0.0
    for successes in range(min_successes, num_trials + 1):
        tail += (
            math.comb(num_trials, successes)
            * (success_prob ** successes)
            * ((1.0 - success_prob) ** (num_trials - successes))
        )
    return clamp(tail, 0.0, 1.0)


def _estimate_bid_statistics(state_features: dict, bid: tuple[int, int]) -> dict:
    own_dice = state_features.get("own_dice") or []
    total_dice = int(state_features.get("total_dice") or 0)
    wild_six_enabled = bool(state_features.get("wild_six_enabled"))
    quantity, face = bid

    if total_dice <= 0 or not own_dice:
        return {
            "known_support": 0,
            "unknown_dice": 0,
            "expected_total": 0.0,
            "std_dev": 0.0,
            "z_score": 0.0,
            "truth_probability": 0.0,
        }

    known_support = _count_face_support(own_dice, face, wild_six_enabled)
    unknown_dice = max(total_dice - len(own_dice), 0)
    per_die_success_prob = 2.0 / 6.0 if (wild_six_enabled and face != 6) else 1.0 / 6.0

    if unknown_dice == 0:
        expected_total = float(known_support)
        std_dev = 0.0
    else:
        expected_total = known_support + unknown_dice * per_die_success_prob
        std_dev = math.sqrt(unknown_dice * per_die_success_prob * (1.0 - per_die_success_prob))

    additional_needed = max(quantity - known_support, 0)
    truth_probability = _binomial_tail_probability(
        num_trials=unknown_dice,
        success_prob=per_die_success_prob,
        min_successes=additional_needed,
    )

    if std_dev > 0:
        z_score = (quantity - expected_total) / std_dev
    elif quantity <= expected_total:
        z_score = -1.0
    else:
        z_score = 3.0

    return {
        "known_support": known_support,
        "unknown_dice": unknown_dice,
        "expected_total": expected_total,
        "std_dev": std_dev,
        "z_score": z_score,
        "truth_probability": truth_probability,
    }


def _score_bid_plausibility(state_features: dict, bid: tuple[int, int]) -> float:
    own_dice = state_features.get("own_dice") or []
    total_dice = int(state_features.get("total_dice") or 0)
    current_bid = state_features.get("current_bid")

    if total_dice <= 0 or not own_dice:
        return 0.0

    stats = _estimate_bid_statistics(state_features, bid)
    p_true = float(stats["truth_probability"])

    reward = 0.0
    if p_true >= 0.60:
        reward += BID_PLAUSIBILITY_BONUS
    elif p_true >= 0.35:
        reward += BID_PLAUSIBILITY_BONUS * 0.5
    elif p_true <= 0.10:
        reward -= BID_PLAUSIBILITY_PENALTY
    elif p_true <= 0.20:
        reward -= BID_PLAUSIBILITY_PENALTY * 0.5

    if current_bid is not None:
        jump = _bid_rank(bid) - _bid_rank(current_bid)
        if jump <= 2:
            reward += 0.01
        elif jump >= 7:
            if p_true < 0.30:
                reward -= 0.03
            else:
                reward += 0.01

    return reward


def _score_challenge_decision(
    state_features: dict,
    chose_liar: bool,
    proposed_bid: tuple[int, int] | None,
) -> float:
    current_bid = state_features.get("current_bid")
    if current_bid is None:
        return 0.0

    stats = _estimate_bid_statistics(state_features, current_bid)
    p_true = float(stats["truth_probability"])
    reward = 0.0

    if not chose_liar and proposed_bid is not None:
        if p_true <= 0.20:
            reward -= PASS_MISSED_CHALLENGE_PENALTY * (
                1.0 + clamp((0.20 - p_true) / 0.20, 0.0, 1.0)
            )
        elif p_true >= 0.55:
            reward += 0.01
    return reward


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class LiarsDiceEnvironment(GameEnvironment):
    def get_config(self) -> GameConfig:
        return GameConfig(
            game_name="liars_dice",
            selected_game="liars_dice",
            mcts_config={
                "opponent": "mcts",
                "mcts_max_simulations": 100,
                "mcts_num_rollouts": 2,
            },
            invalid_action_penalty=INVALID_ACTION_PENALTY,
            shaping_reward_clip=SHAPING_REWARD_CLIP,
            terminal_reward_clip=TERMINAL_REWARD_CLIP,
        )

    def get_curriculum_defaults(self) -> CurriculumDefaults:
        return CurriculumDefaults(
            initial_max_turn=2,
            final_max_turn=20,
            rollouts_per_stage=1280,
            initial_hint_prob=0.0,
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
        # Liar's Dice observations already contain structured legal-action blocks.
        return raw_observation or ""

    def extract_state_features(self, observation: str) -> dict:
        dice: list[int] = []
        dice_match = re.search(r"Your dice:\s*\[([^\]]*)\]", observation)
        if dice_match:
            dice_str = dice_match.group(1).strip()
            if dice_str:
                dice = [int(x.strip()) for x in dice_str.split(",") if x.strip().isdigit()]

        total_dice_match = re.search(r"Total dice in game:\s*(\d+)", observation)
        total_dice = int(total_dice_match.group(1)) if total_dice_match else 0

        current_bid_match = re.search(r'Current bid:\s*"([^"]+)"', observation)
        current_bid = (
            _extract_bid_tuple(current_bid_match.group(1)) if current_bid_match else None
        )

        return {
            "own_dice": dice,
            "total_dice": total_dice,
            "current_bid": current_bid,
            "wild_six_enabled": "wild" in observation.lower() and "6" in observation,
        }

    def parse_action_id(
        self,
        completion_text: str,
        legal_action_map: dict[str, str],
    ) -> str:
        if not legal_action_map:
            return ""

        cleaned = remove_reasoning_tags(completion_text or "")
        if cleaned.endswith("</s>"):
            cleaned = cleaned[:-5]
        if "Action:" in cleaned:
            cleaned = cleaned.split("Action:")[-1].strip()

        # Numeric id match first
        for num in re.findall(r"-?\d+", cleaned):
            if num in legal_action_map:
                return num

        normalized = cleaned.strip().lower()
        for action_id, label in legal_action_map.items():
            if normalized == label.strip().lower():
                return action_id

        if "liar" in normalized:
            for action_id, label in legal_action_map.items():
                if _is_liar_label(label):
                    return action_id

        bid_tuple = _extract_bid_tuple(cleaned)
        if bid_tuple is not None:
            for action_id, label in legal_action_map.items():
                if _extract_bid_tuple(label) == bid_tuple:
                    return action_id

        return ""

    def select_fallback_action(
        self,
        legal_action_map: dict[str, str],
        state_features: dict,
    ) -> str:
        liar_actions = [
            aid for aid, label in legal_action_map.items() if _is_liar_label(label)
        ]
        current_bid = state_features.get("current_bid")
        if liar_actions and current_bid is not None:
            p_true = float(
                _estimate_bid_statistics(state_features, current_bid)["truth_probability"]
            )
            if p_true <= 0.08:
                return liar_actions[0]
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
        chose_liar = _is_liar_label(action_label)
        parsed_bid = _extract_bid_tuple(action_label)

        reward = 0.0
        if parsed_bid is not None:
            reward += _score_bid_plausibility(state_features, parsed_bid)
        reward += _score_challenge_decision(state_features, chose_liar, parsed_bid)
        return reward

    def extract_terminal_reward(self, step_block: dict, observation_text: str) -> float:
        info = step_block.get("info", {}) if isinstance(step_block, dict) else {}

        cumulative_reward = info.get("cumulative_reward")
        if isinstance(cumulative_reward, (int, float)):
            return clamp(float(cumulative_reward), -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)

        your_return_match = re.search(
            r"Your Return:\s*([+-]?\d+(?:\.\d+)?)", observation_text or ""
        )
        if your_return_match:
            return clamp(
                float(your_return_match.group(1)),
                -TERMINAL_REWARD_CLIP,
                TERMINAL_REWARD_CLIP,
            )

        normalized_match = re.search(
            r"Normalized Score:\s*([+-]?\d+(?:\.\d+)?)", observation_text or ""
        )
        result_match = re.search(
            r"Result:\s*(WIN|LOSS|DRAW)", observation_text or "", flags=re.IGNORECASE
        )
        if normalized_match:
            normalized_value = float(normalized_match.group(1))
            if result_match:
                result = result_match.group(1).upper()
                if result == "LOSS":
                    normalized_value = -abs(normalized_value) if normalized_value != 0 else -1.0
                elif result == "WIN":
                    normalized_value = abs(normalized_value) if normalized_value != 0 else 1.0
                else:
                    normalized_value = 0.0
            return clamp(normalized_value, -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)

        step_reward = safe_float(step_block.get("reward", 0.0), default=0.0)
        return clamp(step_reward, -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)
