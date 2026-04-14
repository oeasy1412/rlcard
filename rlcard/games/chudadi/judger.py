from rlcard.games.chudadi.utils import get_legal_actions


class ChuDaDiJudger:
    def __init__(self, np_random, northern_rule=True):
        self.np_random = np_random
        self.northern_rule = northern_rule

    def get_legal_actions(self, hand, last_action, must_contain_card):
        return get_legal_actions(hand, last_action, must_contain_card, self.northern_rule)

    def judge_payoffs(self, players, winner_id, northern_rule=True):
        """Judge payoffs for the game."""
        if winner_id is None:
            return [0 for _ in players]

        scores = []
        for player in players:
            remaining = len(player.current_hand)
            if player.player_id == winner_id:
                score = 0
            else:
                # Northern rule scoring
                if northern_rule:
                    if remaining == 13:
                        # Never played: 39 points (13 * 3)
                        score = 39
                    elif remaining >= 10:
                        # 10-12 cards: score * 2
                        score = remaining * 2
                    else:
                        # Less than 10 cards: normal score
                        score = remaining
                else:
                    # Southern rule scoring
                    # 基础分数按剩余牌数计算
                    score = remaining
                    # 如果有2，扣分翻倍
                    twos = sum(1 for card in player.current_hand if card.rank == "2")
                    if twos > 0:
                        score = score * 2
            scores.append(score)

        total = sum(scores)
        payoffs = []
        for player in players:
            if player.player_id == winner_id:
                payoffs.append(total)
            else:
                payoffs.append(-scores[player.player_id])
        return payoffs
