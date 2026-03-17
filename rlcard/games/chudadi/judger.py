from rlcard.games.chudadi.utils import get_legal_actions


class ChuDaDiJudger:
    def __init__(self, np_random):
        self.np_random = np_random

    def get_legal_actions(self, hand, last_action, must_contain_card):
        return get_legal_actions(hand, last_action, must_contain_card)

    def judge_payoffs(self, players, winner_id):
        if winner_id is None:
            return [0 for _ in players]

        scores = []
        for player in players:
            remaining = len(player.current_hand)
            if player.player_id == winner_id:
                score = 0
            else:
                if remaining == 13:
                    score = 39
                elif remaining >= 8:
                    score = remaining * 2
                else:
                    score = remaining
                twos = sum(1 for card in player.current_hand if card.rank == "2")
                score += twos
            scores.append(score)

        winner = players[winner_id]
        if all(len(p.played_cards) == 0 for p in players if p.player_id != winner_id):
            scores[winner_id] -= 10

        return [-score for score in scores]
