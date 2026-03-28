"""
Opponent modeling module.
Tracks opponent behavior across hands to adapt strategy dynamically.
"""


class OpponentModel:
    """
    Tracks opponent tendencies to enable exploitative play.
    
    Tracked stats:
    - fold_rate: how often opponent folds to aggression (per street)
    - aggression: ratio of (raises) / (calls + raises)  
    - vpip: voluntarily put money in pot preflop
    - redraw_rate: how often opponent uses redraw
    - showdown_rate: how often hands go to showdown
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracking data."""
        # Per-street fold tracking
        self.folds = {0: 0, 3: 0, 4: 0, 5: 0}
        self.fold_opportunities = {0: 0, 3: 0, 4: 0, 5: 0}
        
        # Action tracking
        self.total_raises = 0
        self.total_calls = 0
        self.total_checks = 0
        self.total_folds = 0
        
        # Preflop stats
        self.preflop_raises = 0
        self.preflop_calls = 0
        self.preflop_folds = 0
        self.preflop_hands = 0
        
        # Showdown / redraw
        self.showdowns = 0
        self.total_hands = 0
        self.redraws_used = 0
        self.redraw_opportunities = 0

        # Bet sizing
        self.raise_sizes = []  # (amount, pot_size) tuples

        # Per-hand tracking
        self._current_hand_actions = []
        self._current_street = 0
        self._opponent_raised_preflop = False

    def new_hand(self):
        """Called at the start of each hand."""
        self.total_hands += 1
        self.preflop_hands += 1
        self._current_hand_actions = []
        self._current_street = 0
        self._opponent_raised_preflop = False

    def record_opponent_action(self, action_type, street, amount=0, pot_size=0):
        """
        Record an observed opponent action.
        
        Args:
            action_type: 'fold', 'call', 'check', 'raise', 'redraw'
            street: current street (0, 3, 4, 5)
            amount: raise amount if applicable
            pot_size: current pot size
        """
        self._current_hand_actions.append((action_type, street))
        mapped_street = min(street, 5)
        if mapped_street not in self.folds:
            mapped_street = 0

        if action_type == 'fold':
            self.total_folds += 1
            self.folds[mapped_street] = self.folds.get(mapped_street, 0) + 1
            if street == 0:
                self.preflop_folds += 1

        elif action_type == 'call':
            self.total_calls += 1
            self.fold_opportunities[mapped_street] = self.fold_opportunities.get(mapped_street, 0) + 1
            if street == 0:
                self.preflop_calls += 1

        elif action_type == 'check':
            self.total_checks += 1

        elif action_type == 'raise':
            self.total_raises += 1
            self.fold_opportunities[mapped_street] = self.fold_opportunities.get(mapped_street, 0) + 1
            if street == 0:
                self.preflop_raises += 1
                self._opponent_raised_preflop = True
            if pot_size > 0:
                self.raise_sizes.append((amount, pot_size))

        elif action_type == 'redraw':
            self.redraws_used += 1

    def record_fold_opportunity(self, street):
        """Record that a fold was possible (we bet/raised)."""
        mapped_street = min(street, 5)
        if mapped_street not in self.fold_opportunities:
            mapped_street = 0
        self.fold_opportunities[mapped_street] = self.fold_opportunities.get(mapped_street, 0) + 1

    def record_showdown(self):
        """Record that the hand went to showdown."""
        self.showdowns += 1

    def record_redraw_opportunity(self):
        """Record that redraw was possible for the opponent."""
        self.redraw_opportunities += 1

    # ==== Computed statistics ====

    @property
    def fold_rate(self):
        """Overall fold rate when facing aggression."""
        total_opps = sum(self.fold_opportunities.values())
        if total_opps < 3:
            return 0.35  # Default assumption
        total_folds = sum(self.folds.values())
        return total_folds / total_opps

    def fold_rate_by_street(self, street):
        """Fold rate on a specific street."""
        mapped = min(street, 5)
        if mapped not in self.fold_opportunities:
            return self.fold_rate
        opps = self.fold_opportunities.get(mapped, 0)
        if opps < 2:
            return self.fold_rate  # Not enough data, use overall
        return self.folds.get(mapped, 0) / opps

    @property
    def aggression_factor(self):
        """Aggression factor: raises / (calls + checks)."""
        passive = self.total_calls + self.total_checks
        if passive < 3:
            return 1.0  # Default neutral
        return self.total_raises / passive

    @property
    def vpip(self):
        """Voluntarily put money in pot preflop."""
        if self.preflop_hands < 3:
            return 0.5  # Default
        voluntary = self.preflop_raises + self.preflop_calls
        return voluntary / self.preflop_hands

    @property
    def pfr(self):
        """Preflop raise rate."""
        if self.preflop_hands < 3:
            return 0.3  # Default
        return self.preflop_raises / self.preflop_hands

    @property
    def showdown_rate(self):
        """How often hands reach showdown."""
        if self.total_hands < 3:
            return 0.5
        return self.showdowns / self.total_hands

    @property
    def redraw_rate(self):
        """How often opponent uses their redraw when available."""
        if self.redraw_opportunities < 3:
            return 0.3
        return self.redraws_used / self.redraw_opportunities

    @property
    def avg_raise_size_ratio(self):
        """Average raise size as fraction of pot."""
        if len(self.raise_sizes) < 2:
            return 0.6  # Default
        ratios = [amt / pot for amt, pot in self.raise_sizes if pot > 0]
        if not ratios:
            return 0.6
        return sum(ratios) / len(ratios)

    def get_player_type(self):
        """
        Classify the opponent into an archetype.
        Returns: 'tight-passive', 'tight-aggressive', 'loose-passive', 'loose-aggressive', 'unknown'
        """
        if self.total_hands < 10:
            return 'unknown'

        is_tight = self.vpip < 0.4
        is_aggressive = self.aggression_factor > 1.2

        if is_tight and is_aggressive:
            return 'tight-aggressive'
        if is_tight and not is_aggressive:
            return 'tight-passive'
        if not is_tight and is_aggressive:
            return 'loose-aggressive'
        return 'loose-passive'

    def should_bluff(self, street):
        """
        Determine if we should bluff based on opponent tendencies.
        Returns a bluff frequency (0.0 to 1.0).
        """
        fold_r = self.fold_rate_by_street(street)
        
        # Bluff more against folders
        if fold_r > 0.55:
            return min(0.5, fold_r * 0.7)
        if fold_r > 0.40:
            return 0.25
        if fold_r > 0.25:
            return 0.15
        return 0.05  # Calling station — rarely bluff

    def adjust_value_bet_threshold(self):
        """
        Adjust how thin we value bet based on opponent type.
        Returns equity threshold for value betting (lower = thinner).
        """
        player_type = self.get_player_type()
        if player_type == 'loose-passive':
            return 0.50  # They call a lot — value bet thin
        if player_type == 'tight-passive':
            return 0.60  # They only call with decent hands
        if player_type == 'loose-aggressive':
            return 0.55  # Call downs with medium hands
        if player_type == 'tight-aggressive':
            return 0.65  # They usually have it when they call
        return 0.58  # Default
