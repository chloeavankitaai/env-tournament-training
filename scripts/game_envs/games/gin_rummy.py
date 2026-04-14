"""Gin Rummy game environment.

The most complex concrete game. Game-specific pieces:
- Card parsing (rank/suit/value, runs, sets)
- ``GameState`` dataclass maintained across turns via ``on_episode_start`` /
  ``on_step_complete``
- Deadwood-based episode reward calculator (overrides
  ``compute_episode_reward`` rather than using simple terminal + shaping)
- Optimizer-step hint decay (most other games use rollout-based decay)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from game_envs.base import CurriculumDefaults, GameConfig, GameEnvironment
from game_envs.shared import clamp, remove_reasoning_tags


# ---------------------------------------------------------------------------
# Reward-shaping constants
# ---------------------------------------------------------------------------

INVALID_ACTION_PENALTY = 0.10
SHAPING_REWARD_CLIP = 0.30
TERMINAL_REWARD_CLIP = 2.00  # accommodates +1 win + 0.25 gin bonus

# CurriculumScheduler defaults
INITIAL_MAX_TURN = 50
FINAL_MAX_TURN = 50
INITIAL_HINT_PROB = 0.5
FINAL_HINT_PROB = 0.0
HINT_DECAY_OPTIMIZER_STEPS = 100
INITIAL_MCTS_SIMS = 25
FINAL_MCTS_SIMS = 25


# ---------------------------------------------------------------------------
# Card utilities
# ---------------------------------------------------------------------------

CARD_VALUES = {
    "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "T": 10, "J": 10, "Q": 10, "K": 10,
}

RANK_ORDER = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]


def get_rank(card: str) -> str:
    return card[0]


def get_suit(card: str) -> str:
    return card[1]


def get_value(card: str) -> int:
    return CARD_VALUES[get_rank(card)]


def find_potential_runs(
    hand: list[str], additional_card: Optional[str] = None
) -> list[list[str]]:
    """Find potential runs (2+ consecutive cards same suit)."""
    test_hand = hand.copy()
    if additional_card:
        test_hand.append(additional_card)

    suit_groups: dict[str, list[str]] = {}
    for card in test_hand:
        suit_groups.setdefault(get_suit(card), []).append(card)

    runs: list[list[str]] = []
    for cards in suit_groups.values():
        sorted_cards = sorted(cards, key=lambda c: RANK_ORDER.index(get_rank(c)))
        i = 0
        while i < len(sorted_cards):
            run = [sorted_cards[i]]
            j = i + 1
            while j < len(sorted_cards):
                curr_idx = RANK_ORDER.index(get_rank(sorted_cards[j]))
                prev_idx = RANK_ORDER.index(get_rank(run[-1]))
                if curr_idx == prev_idx + 1:
                    run.append(sorted_cards[j])
                    j += 1
                else:
                    break
            if len(run) >= 2:
                runs.append(run)
            i = j if len(run) > 1 else i + 1
    return runs


def count_complete_runs(hand: list[str]) -> int:
    """Count runs of 3+ consecutive cards same suit."""
    return sum(1 for run in find_potential_runs(hand) if len(run) >= 3)


# ---------------------------------------------------------------------------
# GameState dataclass
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """Snapshot of the gin-rummy state from a single observation."""

    hand: list[str] = field(default_factory=list)
    deadwood: int = 0
    phase: str = "Draw"
    knock_card: int = 10
    upcard: str = "XX"
    stock_size: int = 0
    discard_pile: list[str] = field(default_factory=list)
    player_id: int = 0

    def total_hand_value(self) -> int:
        return sum(get_value(card) for card in self.hand)

    def num_high_cards(self) -> int:
        return sum(1 for card in self.hand if get_value(card) == 10)

    def can_knock(self) -> bool:
        return self.deadwood <= self.knock_card

    def count_pairs(self) -> int:
        rank_counts = Counter(get_rank(card) for card in self.hand)
        return sum(1 for count in rank_counts.values() if count >= 2)

    def count_sets(self) -> int:
        rank_counts = Counter(get_rank(card) for card in self.hand)
        return sum(1 for count in rank_counts.values() if count >= 3)

    def count_runs(self) -> int:
        return count_complete_runs(self.hand)

    def count_potential_runs(self) -> int:
        return sum(1 for run in find_potential_runs(self.hand) if len(run) == 2)


# ---------------------------------------------------------------------------
# Observation parsing helpers
# ---------------------------------------------------------------------------

def _parse_hand_from_observation(observation: str) -> list[str]:
    player_match = re.search(r"You are Player (\d+)", observation)
    player_id = int(player_match.group(1)) if player_match else 0

    player_section_match = re.search(
        rf"Player{player_id}: Deadwood=\d+\n\+-+\+\n(.*?)\n\+-+\+",
        observation,
        re.DOTALL,
    )
    hand: list[str] = []
    if player_section_match:
        card_rows = player_section_match.group(1).strip().split("\n")
        for row in card_rows:
            hand.extend(re.findall(r"([A2-9TJQK][shdc])", row))
    return hand


def _parse_discard_pile(observation: str) -> list[str]:
    discard_match = re.search(r"Discard pile: (.*?)\n", observation)
    if not discard_match:
        return []
    pile_str = discard_match.group(1).strip()
    if not pile_str:
        return []
    if " " in pile_str:
        return pile_str.split()
    return [pile_str[i : i + 2] for i in range(0, len(pile_str), 2)]


def _parse_game_state(observation: str) -> GameState | None:
    """Parse observation into a :class:`GameState` (or ``None`` on failure)."""
    if "Invalid" in observation and "Legal Actions:" not in observation:
        return None

    try:
        player_match = re.search(r"You are Player (\d+)", observation)
        player_id = int(player_match.group(1)) if player_match else 0

        hand = _parse_hand_from_observation(observation)

        deadwood_match = re.search(r"Deadwood=(\d+)", observation)
        deadwood = int(deadwood_match.group(1)) if deadwood_match else 0

        phase_match = re.search(r"Phase: (\w+)", observation)
        phase = phase_match.group(1) if phase_match else "Draw"

        knock_match = re.search(r"Knock card: (\d+)", observation)
        knock_card = int(knock_match.group(1)) if knock_match else 10

        upcard_match = re.search(r"Stock size: \d+\s+Upcard: (\w+)", observation)
        upcard = upcard_match.group(1) if upcard_match else "XX"

        discard_pile = _parse_discard_pile(observation)

        stock_match = re.search(r"Stock size: (\d+)", observation)
        stock_size = int(stock_match.group(1)) if stock_match else 0

        return GameState(
            hand=hand,
            deadwood=deadwood,
            phase=phase,
            knock_card=knock_card,
            upcard=upcard,
            stock_size=stock_size,
            discard_pile=discard_pile,
            player_id=player_id,
        )
    except Exception:
        return None


def _format_gin_rummy_observation(obs_text: str) -> str:
    """Reformat a raw gin_rummy observation to match the evaluation layout."""
    if not obs_text:
        return ""
    if "Invalid action:" in obs_text and "Legal Actions:" in obs_text:
        return obs_text

    state_match = re.search(r"Current State:\n(.*)", obs_text, re.DOTALL)
    if not state_match:
        return obs_text
    state_text = state_match.group(0)

    player_match = re.search(r"You are Player (\d+)", obs_text)
    player_id = int(player_match.group(1)) if player_match else 0

    if "Legal Actions:" not in state_text:
        return obs_text

    current_state_text, legal_action_text = state_text.split("Legal Actions:", 1)
    return (
        current_state_text
        + f"You are Player {player_id}.\nLegal Actions:"
        + legal_action_text
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = (
    "You are playing gin_rummy.\n\n"
    "# Game Rules\n"
    "GIN RUMMY RULES:\n\n"
    "SETUP:\n"
    "- 52-card deck, each player receives 7-10 cards (variant dependent)\n"
    "- Goal: Form MELDS to minimize DEADWOOD (unmelded cards)\n\n"
    "MELDS (Valid Combinations):\n"
    "1. SET: 3+ cards of SAME RANK (e.g., 7\u2660 7\u2665 7\u2663)\n"
    "2. RUN: 3+ CONSECUTIVE cards of SAME SUIT (e.g., 5\u2666 6\u2666 7\u2666)\n"
    "Examples:\n"
    "- Valid runs: A\u2660-2\u2660-3\u2660, 9\u2665-10\u2665-J\u2665-Q\u2665, 10\u2663-J\u2663-Q\u2663-K\u2663\n"
    "- Invalid: K\u2660-A\u2660-2\u2660 (Ace is LOW only, not wraparound)\n\n"
    "CARD NOTATION:\n"
    "- Ranks: A(Ace), 2-9, T(10), J(Jack), Q(Queen), K(King)\n"
    "- Suits: s(spades\u2660), h(hearts\u2665), d(diamonds\u2666), c(clubs\u2663)\n"
    "- Example: 7c = 7 of clubs, Th = 10 of hearts, As = Ace of spades\n\n"
    "GAME PHASES:\n"
    "1. FirstUpcard: Choose to draw first upcard or pass (action IDs: 52=Draw upcard, 54=Pass)\n"
    "2. Draw: Choose to draw from upcard or stock pile (action IDs: 52=Draw upcard, 53=Draw stock)\n"
    "3. Discard: Choose which card to discard (action ID = card's index number, shown in Legal Actions)\n"
    "4. Layoff: After opponent knocks, add cards to their melds or pass (action IDs: card indices or 54=Pass)\n"
    "5. Knock: Declare end of hand when deadwood \u2264 knock_card value\n\n"
    "EACH TURN:\n"
    "1. DRAW phase: Pick from stock pile (53) OR discard pile upcard (52)\n"
    "2. DISCARD phase: Choose ONE card from hand to discard (use card's action ID from Legal Actions)\n\n"
    "KNOCKING:\n"
    "- When deadwood \u2264 knock_card value (8-10), you MAY knock to end hand\n"
    "- Gin: ALL cards form melds (0 deadwood) = 25-point bonus\n\n"
    "SCORING: Winner scores difference in deadwood point values.\n"
    "Card Values: A=1, 2-10=face value, J=11, Q=12, K=13\n\n"
    "IMPORTANT: Always respond with the action ID number ONLY, never card names.\n\n\n"
    "# Output Format\n"
    "You must respond with ONLY the action ID (a single number).\n"
    "Do NOT include descriptions or explanations.\n\n"
    "Examples:\n"
    '- For action "0 -> roll": respond "0"\n'
    '- For action "89 -> a3": respond "89"'
)

STRATEGY_TIPS = (
    "\n\n**Think short and act quickly!**\n\n"
    "# Strategy Tips\n"
    "- Early game: Draw from deck to see more cards\n"
    "- Build runs and sets to reduce deadwood\n"
    "- Track opponent's discards to guess their hand\n"
    "- Knock when you have \u226410 deadwood points and think you're ahead\n"
    "- Go for Gin (0 deadwood) when close for bonus points"
)


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class GinRummyEnvironment(GameEnvironment):
    def get_config(self) -> GameConfig:
        return GameConfig(
            game_name="gin_rummy",
            selected_game="gin_rummy",
            mcts_config={
                "opponent": "mcts",
                "mcts_max_simulations": INITIAL_MCTS_SIMS,
                "mcts_num_rollouts": 1,
            },
            invalid_action_penalty=INVALID_ACTION_PENALTY,
            shaping_reward_clip=SHAPING_REWARD_CLIP,
            terminal_reward_clip=TERMINAL_REWARD_CLIP,
        )

    def get_curriculum_defaults(self) -> CurriculumDefaults:
        return CurriculumDefaults(
            initial_max_turn=INITIAL_MAX_TURN,
            final_max_turn=FINAL_MAX_TURN,
            rollouts_per_stage=1280,
            initial_hint_prob=INITIAL_HINT_PROB,
            final_hint_prob=FINAL_HINT_PROB,
            warmup_rollouts=128,
            hint_decay_mode="optimizer_step",
            hint_decay_optimizer_steps=HINT_DECAY_OPTIMIZER_STEPS,
            mcts_warmup_optimizer_steps=0,
            initial_mcts_sims=INITIAL_MCTS_SIMS,
            final_mcts_sims=FINAL_MCTS_SIMS,
        )

    def get_system_prompt(self, use_hints: bool) -> str:
        prompt = SYSTEM_PROMPT_BASE
        if use_hints:
            prompt += STRATEGY_TIPS
        return prompt

    def extract_and_format_observation(self, raw_observation: str) -> str:
        return _format_gin_rummy_observation(raw_observation or "")

    def extract_state_features(self, observation: str) -> dict:
        game_state = _parse_game_state(observation)
        if game_state is None:
            return {"deadwood": 0, "phase": "Draw"}
        return {
            "deadwood": game_state.deadwood,
            "phase": game_state.phase,
            "can_knock": game_state.can_knock(),
            "hand_size": len(game_state.hand),
        }

    def parse_action_id(
        self,
        completion_text: str,
        legal_action_map: dict[str, str],
    ) -> str:
        # Gin rummy uses pure numeric parsing. The original did not validate
        # the id against the legal map (it let the env reject it), but we do
        # validate here so the engine can fall back when the model hallucinates.
        cleaned = remove_reasoning_tags(completion_text or "")
        cleaned = cleaned.removesuffix("</s>")
        if "Action:" in cleaned:
            cleaned = cleaned.split("Action:")[-1].strip()

        for num in re.findall(r"-?\d+", cleaned):
            if not legal_action_map or num in legal_action_map:
                return num
        return ""

    def select_fallback_action(
        self,
        legal_action_map: dict[str, str],
        state_features: dict,
    ) -> str:
        if not legal_action_map:
            return ""
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
        # Gin rummy has no per-step shaping; invalid penalties are added by
        # the engine automatically, and episode-level reward handles the rest.
        return 0.0

    def extract_terminal_reward(self, step_block: dict, observation_text: str) -> float:
        step_reward = step_block.get("reward", 0.0) if isinstance(step_block, dict) else 0.0
        try:
            return float(step_reward)
        except Exception:
            return 0.0

    # ---- Episode-state hooks ----

    def on_episode_start(self, observation: str) -> Any:
        initial_state = _parse_game_state(observation)
        return {
            "initial": initial_state,
            "latest": initial_state,
            "invalid_count": 0,
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
            episode_state = {"initial": None, "latest": None, "invalid_count": 0}
        if is_invalid:
            episode_state["invalid_count"] = episode_state.get("invalid_count", 0) + 1
            return episode_state
        if done:
            # Final observation of a completed episode does not carry a hand
            # snapshot — keep the most recent live state.
            return episode_state
        new_state = _parse_game_state(observation)
        if new_state is not None:
            episode_state["latest"] = new_state
        return episode_state

    def compute_episode_reward(
        self,
        final_reward: float,
        accumulated_shaping: float,
        done: bool,
        episode_state: Any,
        step_rewards: list[float],
        mode: str = "full",
    ) -> float:
        initial_state: GameState | None = None
        final_state: GameState | None = None
        if isinstance(episode_state, dict):
            initial_state = episode_state.get("initial")
            final_state = episode_state.get("latest")

        # 1. Deadwood improvement (always available, even on truncation)
        if (
            initial_state is not None
            and final_state is not None
            and initial_state.deadwood > 0
        ):
            deadwood_component = (
                initial_state.deadwood - final_state.deadwood
            ) / initial_state.deadwood
        else:
            deadwood_component = 0.0

        # Penalize regressions more harshly
        if deadwood_component < 0.0:
            deadwood_component *= 1.5

        # 2. Terminal bonus / truncation penalty
        if done:
            if final_reward > 0.5:
                terminal = 1.0
                if final_state is not None and final_state.deadwood == 0:
                    terminal += 0.25  # gin bonus
            else:
                terminal = -0.5
        elif final_state is not None:
            terminal = -final_state.deadwood / 100.0
        else:
            terminal = 0.0

        # 3. Invalid action penalty: engine-accumulated shaping is already
        # negative (one -invalid_action_penalty per invalid step). Clip it to
        # -0.3 to match original RewardCalculator behavior.
        invalid_total = min(accumulated_shaping, 0.0)
        invalid_total = max(invalid_total, -0.3)

        return deadwood_component + terminal + invalid_total

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
        record["deadwood"] = state_features.get("deadwood")
        record["phase"] = state_features.get("phase")
        return record
