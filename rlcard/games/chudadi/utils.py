from dataclasses import dataclass

from rlcard.games.base import Card

RANK_ORDER = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2"]
STRAIGHT_RANK_ORDER = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUIT_ORDER = ["D", "C", "H", "S"]

RANK_TO_VALUE = {rank: idx for idx, rank in enumerate(RANK_ORDER)}
STRAIGHT_RANK_TO_INDEX = {rank: idx for idx, rank in enumerate(STRAIGHT_RANK_ORDER)}
SUIT_TO_VALUE = {suit: idx for idx, suit in enumerate(SUIT_ORDER)}

# Feature/action encoding order: [D, C, H, S] x [3..K, A, 2]
DECK_SUIT_ORDER = ["D", "C", "H", "S"]
DECK_RANK_ORDER = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2"]

CARD_ID = {}
ID_TO_CARD = []
for suit_index, suit in enumerate(DECK_SUIT_ORDER):
    for rank_index, rank in enumerate(DECK_RANK_ORDER):
        card = Card(suit, rank)
        card_id = suit_index * len(DECK_RANK_ORDER) + rank_index
        CARD_ID[(suit, rank)] = card_id
        ID_TO_CARD.append(card)

START_CARD = Card("D", "3")

ACTION_TYPES = [
    "single",
    "pair",
    "triple",
    "straight",
    "flush",
    "full_house",
    "four_of_a_kind",
    "straight_flush",
]

ACTION_TYPE_PRIORITY = {name: idx for idx, name in enumerate(ACTION_TYPES)}
MASK_INDICES_CACHE = {}


def card_to_id(card):
    return CARD_ID[(card.suit, card.rank)]


def action_id_to_cards(action_id):
    if action_id == 0:
        return []
    cards = []
    for idx in range(len(ID_TO_CARD)):
        if action_id & (1 << idx):
            cards.append(ID_TO_CARD[idx])
    return cards


def cards_to_action_id(cards):
    action_id = 0
    for card in cards:
        action_id |= 1 << card_to_id(card)
    return action_id


def card_key(card):
    return (RANK_TO_VALUE[card.rank], SUIT_TO_VALUE[card.suit])


def sort_cards(cards):
    return sorted(cards, key=card_key)


def cards_to_str(cards, assume_sorted=False):
    if not cards:
        return "pass"
    if not assume_sorted:
        cards = sort_cards(cards)
    return " ".join(str(card) for card in cards)


@dataclass(frozen=True)
class Action:
    cards: tuple
    action_type: str
    length: int
    key: tuple
    raw: str

    def to_id(self):
        if self.action_type == "pass":
            return 0
        return cards_to_action_id(self.cards)


PASS_ACTION = Action(
    cards=tuple(), action_type="pass", length=0, key=tuple(), raw="pass"
)


def _is_straight(ranks):
    if "2" in ranks:
        return False
    indices = [STRAIGHT_RANK_TO_INDEX[rank] for rank in ranks]
    if len(set(indices)) != len(indices):
        return False
    indices.sort()
    return indices[-1] - indices[0] == len(indices) - 1


def _get_max_card(cards):
    return max(cards, key=card_key)


def make_action(cards):
    if not cards:
        return None
    cards = sort_cards(cards)
    length = len(cards)
    ranks = [card.rank for card in cards]
    suits = [card.suit for card in cards]
    rank_counts = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    unique_ranks = len(rank_counts)
    is_flush = all(suit == suits[0] for suit in suits)
    is_straight = _is_straight(ranks) if length >= 5 else False

    if length == 1:
        max_card = cards[-1]
        return Action(
            tuple(cards),
            "single",
            length,
            card_key(max_card),
            cards_to_str(cards, assume_sorted=True),
        )

    if length == 2 and unique_ranks == 1:
        rank_value = RANK_TO_VALUE[ranks[0]]
        max_suit = SUIT_TO_VALUE[cards[-1].suit]
        return Action(
            tuple(cards),
            "pair",
            length,
            (rank_value, max_suit),
            cards_to_str(cards, assume_sorted=True),
        )

    if length == 3 and unique_ranks == 1:
        rank_value = RANK_TO_VALUE[ranks[0]]
        max_suit = SUIT_TO_VALUE[cards[-1].suit]
        return Action(
            tuple(cards),
            "triple",
            length,
            (rank_value, max_suit),
            cards_to_str(cards, assume_sorted=True),
        )

    if length == 5:
        if is_straight and is_flush:
            max_card = cards[-1]
            return Action(
                tuple(cards),
                "straight_flush",
                length,
                card_key(max_card),
                cards_to_str(cards, assume_sorted=True),
            )

        if length == 5 and sorted(rank_counts.values()) == [2, 3]:
            triple_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
            return Action(
                tuple(cards),
                "full_house",
                length,
                (RANK_TO_VALUE[triple_rank],),
                cards_to_str(cards, assume_sorted=True),
            )

        if length == 5 and sorted(rank_counts.values()) == [1, 4]:
            quad_rank = [rank for rank, count in rank_counts.items() if count == 4][0]
            return Action(
                tuple(cards),
                "four_of_a_kind",
                length,
                (RANK_TO_VALUE[quad_rank],),
                cards_to_str(cards, assume_sorted=True),
            )

        if is_straight:
            max_card = cards[-1]
            return Action(
                tuple(cards),
                "straight",
                length,
                card_key(max_card),
                cards_to_str(cards, assume_sorted=True),
            )

        if is_flush:
            max_card = cards[-1]
            return Action(
                tuple(cards),
                "flush",
                length,
                card_key(max_card),
                cards_to_str(cards, assume_sorted=True),
            )

    return None


def action_to_feature_meta(cards):
    if not cards:
        return "none", None, None
    action = make_action(cards)
    if action is None:
        return "none", None, None

    action_type = action.action_type
    main_rank = None
    kicker_rank = None

    if action_type in ("single", "pair", "triple"):
        main_rank = action.cards[0].rank
    elif action_type in ("straight", "flush", "straight_flush"):
        max_card = _get_max_card(action.cards)
        main_rank = max_card.rank
        if len(action.cards) >= 2:
            kicker_rank = action.cards[-2].rank
    elif action_type == "full_house":
        rank_counts = {}
        for card in action.cards:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
        for rank, count in rank_counts.items():
            if count == 3:
                main_rank = rank
            elif count == 2:
                kicker_rank = rank
    elif action_type == "four_of_a_kind":
        rank_counts = {}
        for card in action.cards:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
        for rank, count in rank_counts.items():
            if count == 4:
                main_rank = rank
            elif count == 1:
                kicker_rank = rank

    main_index = RANK_TO_VALUE[main_rank] if main_rank is not None else None
    kicker_index = RANK_TO_VALUE[kicker_rank] if kicker_rank is not None else None
    return action_type, main_index, kicker_index


def can_beat(action, last_action, northern_rule=True):
    if last_action is None:
        return True

    # 南方规则：铁支和同花顺有特殊压制能力，其余牌型必须同类型比较
    if not northern_rule:
        # 同花顺可压任意牌型
        if action.action_type == "straight_flush":
            if last_action.action_type == "straight_flush":
                return action.key > last_action.key
            else:
                return True
        # 铁支可压除同花顺外的任意牌型
        if action.action_type == "four_of_a_kind":
            if last_action.action_type == "straight_flush":
                return False
            elif last_action.action_type == "four_of_a_kind":
                return action.key > last_action.key
            else:
                return True
        # 南方规则：普通牌型必须同类型比较
        if action.action_type != last_action.action_type:
            return False
        return action.key > last_action.key

    # 北方规则：五张牌型可以相互压制
    five_card_types = (
        "straight",
        "flush",
        "full_house",
        "four_of_a_kind",
        "straight_flush",
    )
    is_five_card_action = action.action_type in five_card_types
    is_five_card_last = last_action.action_type in five_card_types

    if is_five_card_action and is_five_card_last:
        # 五张牌型可以相互压制，按类型优先级
        action_priority = ACTION_TYPE_PRIORITY.get(action.action_type, -1)
        last_priority = ACTION_TYPE_PRIORITY.get(last_action.action_type, -1)
        if action_priority > last_priority:
            return True
        elif action_priority < last_priority:
            return False
        else:
            # 相同类型，比较key
            return action.key > last_action.key

    # 其他牌型必须同类型才能比较
    if action.action_type != last_action.action_type:
        return False
    return action.key > last_action.key


def _get_mask_indices(num_cards):
    cache = MASK_INDICES_CACHE.get(num_cards)
    if cache is not None:
        return cache
    cache = [[] for _ in range(1 << num_cards)]
    for mask in range(1, 1 << num_cards):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev = mask ^ lsb
        cache[mask] = cache[prev] + [idx]
    MASK_INDICES_CACHE[num_cards] = cache
    return cache


def _generate_valid_actions(hand):
    cards = sort_cards(hand)
    num_cards = len(cards)
    actions = []
    mask_indices = _get_mask_indices(num_cards)
    for mask in range(1, 1 << num_cards):
        indices = mask_indices[mask]
        length = len(indices)
        if length == 1:
            card = cards[indices[0]]
            actions.append(Action((card,), "single", 1, card_key(card), str(card)))
            continue
        if length == 2:
            card1 = cards[indices[0]]
            card2 = cards[indices[1]]
            if card1.rank != card2.rank:
                continue
            max_suit = SUIT_TO_VALUE[card2.suit]
            rank_value = RANK_TO_VALUE[card1.rank]
            actions.append(
                Action(
                    (card1, card2),
                    "pair",
                    2,
                    (rank_value, max_suit),
                    cards_to_str((card1, card2), assume_sorted=True),
                )
            )
            continue
        if length == 3:
            card1 = cards[indices[0]]
            card2 = cards[indices[1]]
            card3 = cards[indices[2]]
            if card1.rank != card2.rank or card2.rank != card3.rank:
                continue
            max_suit = SUIT_TO_VALUE[card3.suit]
            rank_value = RANK_TO_VALUE[card1.rank]
            actions.append(
                Action(
                    (card1, card2, card3),
                    "triple",
                    3,
                    (rank_value, max_suit),
                    cards_to_str((card1, card2, card3), assume_sorted=True),
                )
            )
            continue
        if length < 5:
            continue
        if length > 5:
            continue
        subset = [cards[i] for i in indices]
        action = make_action(subset)
        if action is not None:
            actions.append(action)
    return actions


def get_legal_actions(hand, last_action, must_contain_card=False, northern_rule=True):
    """Get legal actions for the player.
    Args:
        hand: Player's current hand
        last_action: The last action played
        must_contain_card: Whether the action must contain the start card (D3)
        northern_rule: If True, apply northern rule (must play if can beat same type)
    Returns:
        List of legal Action objects
    """
    actions = _generate_valid_actions(hand)
    if must_contain_card:
        actions = [action for action in actions if START_CARD in action.cards]

    if last_action is not None:
        beatable_actions = [
            action for action in actions if can_beat(action, last_action, northern_rule)
        ]

        if northern_rule:
            # Northern rule: must play same type if can beat
            five_card_types = (
                "straight",
                "flush",
                "full_house",
                "four_of_a_kind",
                "straight_flush",
            )
            if last_action.action_type in five_card_types:
                # For five-card hands, all five-card types are considered the same category
                has_same_type_beat = any(
                    action.action_type in five_card_types for action in beatable_actions
                )
                if has_same_type_beat:
                    actions = [
                        action
                        for action in beatable_actions
                        if action.action_type in five_card_types
                    ]
                else:
                    actions = (
                        beatable_actions + [PASS_ACTION]
                        if beatable_actions
                        else [PASS_ACTION]
                    )
            else:
                has_same_type_beat = any(
                    action.action_type == last_action.action_type
                    for action in beatable_actions
                )
                if has_same_type_beat:
                    actions = [
                        action
                        for action in beatable_actions
                        if action.action_type == last_action.action_type
                    ]
                else:
                    actions = (
                        beatable_actions + [PASS_ACTION]
                        if beatable_actions
                        else [PASS_ACTION]
                    )
        else:
            # Southern rule: any beating action is valid (including special five-card types)
            if beatable_actions:
                actions = beatable_actions + [PASS_ACTION]
            else:
                actions = [PASS_ACTION]

    actions.sort(
        key=lambda action: (
            ACTION_TYPE_PRIORITY.get(action.action_type, 99),
            action.length,
            action.key,
            [card_key(card) for card in action.cards],
        )
    )
    return actions
