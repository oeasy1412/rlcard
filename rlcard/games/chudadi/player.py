from rlcard.games.chudadi.utils import sort_cards


class ChuDaDiPlayer:
    def __init__(self, player_id):
        self.player_id = player_id
        self.current_hand = []
        self.played_cards = []
        self.has_played = False

    def set_current_hand(self, cards):
        self.current_hand = sort_cards(cards)
        self.played_cards = []
        self.has_played = False

    def play_cards(self, cards):
        for card in cards:
            self.current_hand.remove(card)
            self.played_cards.append(card)
        self.current_hand = sort_cards(self.current_hand)
        if cards:
            self.has_played = True
