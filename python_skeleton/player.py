'''
Aggressive poker bot with Monte Carlo evaluation, opponent modeling,
and smart situational redraw logic.
'''
import random
import time

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, RedrawAction
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

from hand_evaluator import (
    evaluate_hand_strength,
    get_preflop_category,
    smart_redraw_decision,
    _card_rank,
)
from opponent_model import OpponentModel


class Player(Bot):
    '''
    Aggressive poker bot with Monte Carlo hand evaluation,
    situational redraw, and opponent exploitation.
    '''

    def __init__(self):
        self.opponent = OpponentModel()
        self.round_num = 0
        self.my_bankroll = 0
        self.time_remaining = 180.0
        self.hand_start_time = None

        # Per-hand state
        self.my_hand = []
        self.active = 0
        self.is_button = False  # True if we're the small blind / button
        self.used_redraw = False
        self.street_strengths = {}  # Cache: street -> strength

    def _time_per_action(self):
        """Estimate how much time we can spend per action."""
        hands_left = max(1, 300 - self.round_num)
        # Reserve some buffer
        safe_time = max(0.5, self.time_remaining - 5.0)
        # Roughly 3 actions per hand
        return safe_time / (hands_left * 3)

    def _get_sim_count(self, base=300):
        """Dynamically adjust simulation count based on time budget."""
        time_budget = self._time_per_action()
        if time_budget > 0.3:
            return base
        elif time_budget > 0.15:
            return base // 2
        elif time_budget > 0.08:
            return base // 4
        else:
            return max(30, base // 8)

    def _get_strength(self, round_state, force_recalc=False):
        """Get hand strength, using cache when possible."""
        street = round_state.street
        board_key = tuple(round_state.board)
        hand_key = tuple(round_state.hands[self.active])
        cache_key = (street, board_key, hand_key)
        
        if not force_recalc and cache_key in self.street_strengths:
            return self.street_strengths[cache_key]
        
        sims = self._get_sim_count(300)
        strength = evaluate_hand_strength(
            round_state.hands[self.active],
            round_state.board,
            num_simulations=sims,
        )
        self.street_strengths[cache_key] = strength
        return strength

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts.
        '''
        self.round_num = game_state.round_num
        self.my_bankroll = game_state.bankroll
        self.time_remaining = game_state.game_clock
        self.active = active
        self.my_hand = list(round_state.hands[active])
        self.is_button = (active == 0)  # Player 0 is SB/button preflop
        self.used_redraw = False
        self.street_strengths = {}
        self.hand_start_time = time.perf_counter()

        self.opponent.new_hand()

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Update opponent model.
        '''
        self.time_remaining = game_state.game_clock
        
        # Check if it went to showdown (not a fold)
        prev = terminal_state.previous_state
        if prev is not None:
            # If deltas are non-zero and neither player folded, it's a showdown
            delta = terminal_state.deltas[active]
            # Heuristic: if the game reached river or both hands visible, it's a showdown
            if hasattr(prev, 'street') and prev.street >= 5:
                self.opponent.record_showdown()

    def _pot_size(self, round_state):
        """Calculate current pot size."""
        return 2 * STARTING_STACK - round_state.stacks[0] - round_state.stacks[1]

    def _continue_cost(self, round_state):
        """Cost to continue in the hand."""
        return round_state.pips[1 - self.active] - round_state.pips[self.active]

    def _pot_odds(self, round_state):
        """Calculate pot odds (ratio of call cost to pot + call)."""
        cost = self._continue_cost(round_state)
        if cost <= 0:
            return 0.0
        pot = self._pot_size(round_state)
        return cost / (pot + cost)

    def _raise_amount(self, round_state, fraction_of_pot):
        """Calculate raise amount as a fraction of the pot."""
        if RaiseAction not in round_state.legal_actions():
            return None
        min_raise, max_raise = round_state.raise_bounds()
        pot = self._pot_size(round_state)
        cost = self._continue_cost(round_state)
        
        # Raise TO amount = current pip + cost + (fraction * (pot + cost))
        desired = round_state.pips[self.active] + cost + int(fraction_of_pot * (pot + cost))
        desired = max(min_raise, min(desired, max_raise))
        return desired

    def _get_preflop_action(self, game_state, round_state):
        """Aggressive preflop strategy."""
        legal = round_state.legal_actions()
        category = get_preflop_category(round_state.hands[self.active])
        cost = self._continue_cost(round_state)
        pot = self._pot_size(round_state)

        if category == 'premium':
            # Big raise or re-raise
            if RaiseAction in legal:
                if cost > 0:
                    # Re-raise: 3x their raise
                    amount = self._raise_amount(round_state, 1.0)
                else:
                    # Open raise: 3-4x BB
                    amount = self._raise_amount(round_state, 2.0)
                if amount:
                    return RaiseAction(amount)
            return CallAction() if CallAction in legal else CheckAction()

        elif category == 'strong':
            if RaiseAction in legal:
                if cost == 0:
                    # Open raise
                    amount = self._raise_amount(round_state, 1.5)
                elif cost <= 15:
                    # Call moderate raises, re-raise small ones
                    amount = self._raise_amount(round_state, 0.8)
                else:
                    return CallAction() if CallAction in legal else CheckAction()
                if amount:
                    return RaiseAction(amount)
            return CallAction() if CallAction in legal else CheckAction()

        elif category == 'medium':
            if cost == 0:
                # Open raise with medium hands when in position
                if RaiseAction in legal and (self.is_button or random.random() < 0.6):
                    amount = self._raise_amount(round_state, 1.0)
                    if amount:
                        return RaiseAction(amount)
                return CheckAction() if CheckAction in legal else CallAction()
            elif cost <= 8:
                return CallAction() if CallAction in legal else FoldAction()
            else:
                # Fold medium hands to big raises
                if random.random() < 0.3:
                    return CallAction() if CallAction in legal else FoldAction()
                return FoldAction()

        elif category == 'playable':
            if cost == 0:
                # Sometimes open with playable hands
                if RaiseAction in legal and self.is_button and random.random() < 0.45:
                    amount = self._raise_amount(round_state, 0.8)
                    if amount:
                        return RaiseAction(amount)
                return CheckAction() if CheckAction in legal else CallAction()
            elif cost <= 4:
                return CallAction() if CallAction in legal else FoldAction()
            else:
                return FoldAction()

        else:  # weak
            if cost == 0:
                # Steal occasionally from button
                if self.is_button and RaiseAction in legal and random.random() < 0.30:
                    bluff_freq = self.opponent.should_bluff(0)
                    if random.random() < bluff_freq + 0.15:
                        amount = self._raise_amount(round_state, 1.0)
                        if amount:
                            return RaiseAction(amount)
                return CheckAction() if CheckAction in legal else FoldAction()
            else:
                return FoldAction()

    def _get_postflop_action(self, game_state, round_state):
        """Aggressive postflop strategy driven by Monte Carlo equity."""
        legal = round_state.legal_actions()
        strength = self._get_strength(round_state)
        cost = self._continue_cost(round_state)
        pot = self._pot_size(round_state)
        pot_odds = self._pot_odds(round_state)
        street = round_state.street

        # Bluff parameters from opponent model
        bluff_freq = self.opponent.should_bluff(street)
        value_threshold = self.opponent.adjust_value_bet_threshold()

        # ---- STRONG HAND (nuts or near-nuts) ----
        if strength >= 0.85:
            if RaiseAction in legal:
                # Large value bet / raise
                amount = self._raise_amount(round_state, 0.8 + random.random() * 0.3)
                if amount:
                    return RaiseAction(amount)
            return CallAction() if CallAction in legal else CheckAction()

        # ---- GOOD HAND ----
        if strength >= 0.70:
            if cost == 0 and RaiseAction in legal:
                # Value bet
                amount = self._raise_amount(round_state, 0.55 + random.random() * 0.25)
                if amount:
                    return RaiseAction(amount)
            elif cost > 0:
                if RaiseAction in legal and cost < pot * 0.5 and random.random() < 0.4:
                    # Re-raise good hands sometimes
                    amount = self._raise_amount(round_state, 0.6)
                    if amount:
                        return RaiseAction(amount)
                return CallAction() if CallAction in legal else CheckAction()
            return CheckAction() if CheckAction in legal else CallAction()

        # ---- DECENT HAND ----
        if strength >= value_threshold:
            if cost == 0:
                if RaiseAction in legal and random.random() < 0.5:
                    amount = self._raise_amount(round_state, 0.35 + random.random() * 0.2)
                    if amount:
                        return RaiseAction(amount)
                return CheckAction() if CheckAction in legal else CallAction()
            elif strength > pot_odds + 0.05:
                return CallAction() if CallAction in legal else FoldAction()
            else:
                return FoldAction()

        # ---- MARGINAL HAND ----
        if strength >= 0.35:
            if cost == 0:
                # Check, occasionally bluff
                if RaiseAction in legal and random.random() < bluff_freq * 0.6:
                    amount = self._raise_amount(round_state, 0.4 + random.random() * 0.2)
                    if amount:
                        return RaiseAction(amount)
                return CheckAction() if CheckAction in legal else CallAction()
            elif strength > pot_odds:
                # Pot odds justify a call
                return CallAction() if CallAction in legal else FoldAction()
            else:
                # Small bet? Call anyway sometimes
                if cost <= BIG_BLIND * 3 and random.random() < 0.3:
                    return CallAction() if CallAction in legal else FoldAction()
                return FoldAction()

        # ---- WEAK HAND ----
        if cost == 0:
            # Check, or bluff occasionally
            if RaiseAction in legal and random.random() < bluff_freq:
                amount = self._raise_amount(round_state, 0.5 + random.random() * 0.3)
                if amount:
                    return RaiseAction(amount)
            return CheckAction() if CheckAction in legal else FoldAction()
        else:
            # Only call if pot odds are extremely favorable
            if strength > pot_odds and cost <= BIG_BLIND * 2:
                return CallAction() if CallAction in legal else FoldAction()
            return FoldAction()

    def _consider_redraw(self, game_state, round_state):
        """
        Decide whether to redraw and what to target.
        Returns (target_type, target_index) or None.
        """
        if self.used_redraw:
            return None
        if RedrawAction not in round_state.legal_actions():
            return None
        if round_state.street < 3 or round_state.street >= 5:
            return None

        # Use smart situational redraw
        sims = self._get_sim_count(150)
        result = smart_redraw_decision(
            round_state.hands[self.active],
            round_state.board,
            self.active,
            round_state.redraws_used,
            round_state.street,
            num_simulations=sims,
        )
        return result

    def get_action(self, game_state, round_state, active):
        '''
        Main decision function. Called every time the engine needs an action.
        '''
        self.active = active
        self.time_remaining = game_state.game_clock
        legal = round_state.legal_actions()

        # ---- Consider redraw first ----
        redraw_target = self._consider_redraw(game_state, round_state)

        # ---- Determine the betting action ----
        if round_state.street == 0:
            betting_action = self._get_preflop_action(game_state, round_state)
        else:
            betting_action = self._get_postflop_action(game_state, round_state)

        # ---- Wrap with redraw if applicable ----
        if redraw_target is not None:
            target_type, target_index = redraw_target
            self.used_redraw = True
            
            # Validate the inner betting action is legal
            basic_legal = set(legal) - {RedrawAction}
            if type(betting_action) not in basic_legal:
                # Fall back to check or call
                if CheckAction in basic_legal:
                    betting_action = CheckAction()
                elif CallAction in basic_legal:
                    betting_action = CallAction()
                else:
                    betting_action = FoldAction()
            
            return RedrawAction(target_type, target_index, betting_action)

        # ---- Validate action ----
        if type(betting_action) not in legal:
            if CheckAction in legal:
                return CheckAction()
            if CallAction in legal:
                return CallAction()
            return FoldAction()

        return betting_action


if __name__ == '__main__':
    run_bot(Player(), parse_args())
