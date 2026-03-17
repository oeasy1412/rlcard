from rlcard.utils import init_standard_deck


class ChuDaDiDealer:
    def __init__(self, np_random):
        self.np_random = np_random
        self.deck = init_standard_deck()

    def shuffle(self):
        self.np_random.shuffle(self.deck)

    def deal_cards(self, players):
        hand_size = len(self.deck) // len(players)
        for index, player in enumerate(players):
            start = index * hand_size
            end = (index + 1) * hand_size
            player.set_current_hand(self.deck[start:end])
