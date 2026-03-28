"""
Monte Carlo hand strength evaluator using pkrbot.
Handles both current hand evaluation and redraw value estimation.

Cards in the game are strings like 'As', 'Kd', '5h'.
pkrbot.evaluate() requires pkrbot.Card objects.
"""
import random
import pkrbot

# Full deck of 52 cards as strings
RANKS = '23456789TJQKA'
SUITS = 'shdc'
FULL_DECK = [r + s for r in RANKS for s in SUITS]

# Preflop hand rankings
RANK_VALUES = {r: i for i, r in enumerate(RANKS)}

# Premium preflop hands (as sorted rank tuples)
PREMIUM_HANDS = {
    ('A', 'A'), ('K', 'K'), ('Q', 'Q'), ('A', 'K'),
}
STRONG_HANDS = {
    ('J', 'J'), ('T', 'T'), ('9', '9'), ('A', 'Q'), ('A', 'J'),
    ('K', 'Q'),
}

# Pre-build Card object cache for speed (avoid repeated Card() calls)
_CARD_CACHE = {}
def _get_card(s):
    """Get or create a pkrbot.Card from a string. Caches for reuse."""
    if s not in _CARD_CACHE:
        _CARD_CACHE[s] = pkrbot.Card(s)
    return _CARD_CACHE[s]

# Pre-populate cache
for _cs in FULL_DECK:
    _get_card(_cs)


def _card_rank(card):
    """Returns numeric rank value (0=2, 12=A) for a card string like 'Ah'."""
    if not card or card == '??' or len(card) < 2:
        return -1
    return RANK_VALUES.get(card[0], -1)


def _card_suit(card):
    """Returns suit character for a card string."""
    if not card or card == '??' or len(card) < 2:
        return '?'
    return card[1]


def _is_suited(hand):
    """Check if two hole cards are suited."""
    if len(hand) < 2:
        return False
    return _card_suit(hand[0]) == _card_suit(hand[1])


def _is_pair(hand):
    """Check if two hole cards form a pair."""
    if len(hand) < 2:
        return False
    return hand[0][0] == hand[1][0]


def _to_str(card):
    """Convert card (str or Card object) to string."""
    return str(card)


def get_preflop_category(hand):
    """
    Classify a preflop hand into categories.
    Returns: 'premium', 'strong', 'medium', 'playable', 'weak'
    """
    str_hand = [_to_str(c) for c in hand]
    if len(str_hand) < 2 or '??' in str_hand:
        return 'weak'

    r0, r1 = _card_rank(str_hand[0]), _card_rank(str_hand[1])
    high_r, low_r = max(r0, r1), min(r0, r1)
    high_c = str_hand[0][0] if r0 >= r1 else str_hand[1][0]
    low_c = str_hand[1][0] if r0 >= r1 else str_hand[0][0]
    suited = _is_suited(str_hand)

    # Pairs
    if _is_pair(str_hand):
        if high_r >= RANK_VALUES['Q']:  # QQ+
            return 'premium'
        if high_r >= RANK_VALUES['9']:  # 99+
            return 'strong'
        if high_r >= RANK_VALUES['5']:  # 55+
            return 'medium'
        return 'playable'

    key = (high_c, low_c)
    if key in PREMIUM_HANDS:
        return 'premium'
    if key in STRONG_HANDS:
        return 'strong' if suited else 'medium'

    # Suited connectors / suited aces
    if suited:
        if high_c == 'A':
            return 'medium'  # Suited ace
        gap = high_r - low_r
        if gap == 1 and low_r >= RANK_VALUES['5']:  # Suited connector 56s+
            return 'medium'
        if gap <= 2 and low_r >= RANK_VALUES['7']:  # Suited one-gapper
            return 'playable'

    # High cards
    if high_r >= RANK_VALUES['K'] and low_r >= RANK_VALUES['T']:
        return 'playable'

    # Connected cards
    if abs(r0 - r1) == 1 and min(r0, r1) >= RANK_VALUES['8']:
        return 'playable'

    return 'weak'


def _get_available_cards(known_cards):
    """Returns list of card strings not in known_cards. Filters out '??' placeholders."""
    known_set = set(_to_str(c) for c in known_cards if c and _to_str(c) != '??')
    return [c for c in FULL_DECK if c not in known_set]


def _evaluate_7(cards_strs):
    """Evaluate a 7-card hand from string card representations. Returns score (higher = better)."""
    card_objs = [_get_card(s) for s in cards_strs]
    return pkrbot.evaluate(card_objs)


def evaluate_hand_strength(my_hand, board, num_simulations=300):
    """
    Monte Carlo hand strength estimation.
    Simulates random opponent hands and remaining board cards,
    returns win probability (0.0 to 1.0).
    
    Args:
        my_hand: list of 2 card strings (e.g., ['Ah', 'Kd'])
        board: list of 0-5 card strings
        num_simulations: number of Monte Carlo iterations
    
    Returns:
        float: estimated win probability (0.0 to 1.0)
    """
    # Convert to strings and filter unknowns
    str_hand = [_to_str(c) for c in my_hand]
    str_board = [_to_str(c) for c in board]
    
    valid_hand = [c for c in str_hand if c and c != '??']
    valid_board = [c for c in str_board if c and c != '??']

    if len(valid_hand) < 2:
        return 0.5  # Can't evaluate with unknown cards

    known_cards = valid_hand + valid_board
    available = _get_available_cards(known_cards)

    if len(available) < 2:
        return 0.5

    wins = 0
    ties = 0
    total = 0
    board_cards_needed = 5 - len(valid_board)
    needed = 2 + board_cards_needed

    if len(available) < needed:
        return 0.5

    for _ in range(num_simulations):
        sampled = random.sample(available, needed)
        opp_hand = sampled[:2]
        remaining_board = sampled[2:]
        full_board = valid_board + remaining_board

        try:
            my_score = _evaluate_7(valid_hand + full_board)
            opp_score = _evaluate_7(opp_hand + full_board)

            if my_score > opp_score:
                wins += 1
            elif my_score == opp_score:
                ties += 1
            total += 1
        except Exception:
            continue

    if total == 0:
        return 0.5

    return (wins + 0.5 * ties) / total


def smart_redraw_decision(my_hand, board, active, redraws_used, street, num_simulations=200):
    """
    Situational redraw:
    - If hole cards are "good" → try redrawing a board card to improve the board
    - If hole cards are "bad" → redraw weakest hole card
    
    Only considers redraws on flop (street=3) or turn (street=4).
    
    Returns:
        tuple: (target_type, target_index) or None if no redraw should be done
    """
    if redraws_used[active] or street >= 5 or street < 3:
        return None

    str_hand = [_to_str(c) for c in my_hand]
    str_board = [_to_str(c) for c in board]
    
    valid_hand = [c for c in str_hand if c and c != '??']
    valid_board = [c for c in str_board if c and c != '??']

    if len(valid_hand) < 2 or len(valid_board) < 3:
        return None

    # Evaluate current strength
    current_strength = evaluate_hand_strength(str_hand, str_board, num_simulations=num_simulations)
    
    # If already very strong, don't bother redrawing
    if current_strength >= 0.80:
        return None

    known_cards = valid_hand + valid_board
    available = _get_available_cards(known_cards)

    if len(available) < 3:
        return None

    # Determine if hole cards are "good" or "bad"
    preflop_cat = get_preflop_category(str_hand)
    hole_cards_good = preflop_cat in ('premium', 'strong', 'medium')

    best_target = None
    best_avg_strength = current_strength
    min_improvement = 0.05  # Must improve by at least 5% equity

    if hole_cards_good and current_strength >= 0.40:
        # Good hole cards → try redrawing board cards
        board_limit = {3: 3, 4: 4}.get(street, 0)
        num_samples = min(len(available), 35)
        
        for idx in range(min(board_limit, len(valid_board))):
            if str_board[idx] == '??':
                continue
            
            avail = [c for c in available if c != str_board[idx]]
            if len(avail) < 1:
                continue
            sample = random.sample(avail, min(len(avail), num_samples))
            total = 0.0
            count = 0
            for nc in sample:
                new_board = list(str_board)
                new_board[idx] = nc
                strength = evaluate_hand_strength(str_hand, new_board, num_simulations=60)
                total += strength
                count += 1
            
            if count > 0:
                avg = total / count
                if avg > best_avg_strength + min_improvement:
                    best_avg_strength = avg
                    best_target = ('board', idx)
    else:
        # Bad/marginal hole cards → redraw weakest hole card
        r0 = _card_rank(str_hand[0])
        r1 = _card_rank(str_hand[1])
        weakest_idx = 0 if r0 <= r1 else 1
        
        num_samples = min(len(available), 45)
        sample = random.sample(available, num_samples)
        total = 0.0
        count = 0
        for nc in sample:
            new_hand = list(str_hand)
            new_hand[weakest_idx] = nc
            strength = evaluate_hand_strength(new_hand, str_board, num_simulations=60)
            total += strength
            count += 1
        
        if count > 0:
            avg = total / count
            if avg > best_avg_strength + min_improvement:
                best_target = ('hole', weakest_idx)

    return best_target
