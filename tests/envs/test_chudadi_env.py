import unittest

import rlcard
from rlcard.games.base import Card
from rlcard.games.chudadi.judger import ChuDaDiJudger
from rlcard.games.chudadi.player import ChuDaDiPlayer
from rlcard.games.chudadi.utils import (
    PASS_ACTION,
    RANK_TO_VALUE,
    START_CARD,
    can_beat,
    card_to_id,
    cards_to_action_id,
    get_legal_actions,
    make_action,
)


class TestChudadiEnv(unittest.TestCase):
    def setUp(self):
        self.env = rlcard.make("chudadi")

    @staticmethod
    def _make_cards(ranks):
        suits = ["D", "C", "H", "S"]
        return [Card(suits[i % len(suits)], rank) for i, rank in enumerate(ranks)]

    def _get_feature(self, action_cards, extra_cards=None, hand_override=None):
        if hand_override is not None:
            hand = hand_override
        else:
            hand = list(action_cards) + (extra_cards or [])
        state = {"raw_obs": {"current_hand": hand}}
        action_id = cards_to_action_id(action_cards) if action_cards else 0
        feature = self.env.get_action_feature(action_id, state)
        return feature, hand

    def _assert_one_hot(self, segment, expected_index=None):
        if expected_index is None:
            self.assertEqual(int(segment.sum()), 0)
            return
        self.assertEqual(int(segment.sum()), 1)
        self.assertEqual(int(segment[expected_index]), 1)

    def _assert_action_future_bits(self, feature, action_cards, hand):
        action_set = {(card.suit, card.rank) for card in action_cards}
        action_bits = feature[:52]
        future_bits = feature[52:104]
        self.assertEqual(int(action_bits.sum()), len(action_cards))
        for card in action_cards:
            idx = card_to_id(card)
            self.assertEqual(int(action_bits[idx]), 1)
            self.assertEqual(int(future_bits[idx]), 0)
        extra = None
        for card in hand:
            if (card.suit, card.rank) not in action_set:
                extra = card
                break
        if extra is not None:
            idx = card_to_id(extra)
            self.assertEqual(int(action_bits[idx]), 0)
            self.assertEqual(int(future_bits[idx]), 1)

    def test_action_feature_meta(self):
        """验证各牌型的动作特征维度、action/future_hand bits、action_type、main_rank、kicker_rank。"""
        extra = [Card("C", "2")]
        cases = [
            (
                "single",
                [Card("S", "A")],
                1,
                RANK_TO_VALUE["A"],
                None,
            ),
            (
                "pair",
                [Card("D", "5"), Card("S", "5")],
                2,
                RANK_TO_VALUE["5"],
                None,
            ),
            (
                "triple",
                [Card("D", "8"), Card("H", "8"), Card("S", "8")],
                3,
                RANK_TO_VALUE["8"],
                None,
            ),
            (
                "straight",
                [
                    Card("D", "7"),
                    Card("C", "8"),
                    Card("H", "9"),
                    Card("S", "T"),
                    Card("D", "J"),
                ],
                4,
                RANK_TO_VALUE["J"],
                RANK_TO_VALUE["T"],
            ),
            (
                "flush",
                [
                    Card("H", "3"),
                    Card("H", "5"),
                    Card("H", "7"),
                    Card("H", "9"),
                    Card("H", "J"),
                ],
                5,
                RANK_TO_VALUE["J"],
                RANK_TO_VALUE["9"],
            ),
            (
                "full_house",
                [
                    Card("D", "K"),
                    Card("C", "K"),
                    Card("H", "K"),
                    Card("S", "4"),
                    Card("H", "4"),
                ],
                6,
                RANK_TO_VALUE["K"],
                RANK_TO_VALUE["4"],
            ),
            (
                "four_of_a_kind",
                [
                    Card("D", "Q"),
                    Card("C", "Q"),
                    Card("H", "Q"),
                    Card("S", "Q"),
                    Card("D", "3"),
                ],
                7,
                RANK_TO_VALUE["Q"],
                RANK_TO_VALUE["3"],
            ),
            (
                "straight_flush",
                [
                    Card("S", "9"),
                    Card("S", "T"),
                    Card("S", "J"),
                    Card("S", "Q"),
                    Card("S", "K"),
                ],
                8,
                RANK_TO_VALUE["K"],
                RANK_TO_VALUE["Q"],
            ),
        ]

        for _, cards, type_index, main_index, kicker_index in cases:
            feature, hand = self._get_feature(cards, extra_cards=extra)
            self.assertEqual(feature.size, 139)
            self._assert_action_future_bits(feature, cards, hand)
            self._assert_one_hot(feature[104:113], type_index)
            self._assert_one_hot(feature[113:126], main_index)
            self._assert_one_hot(feature[126:139], kicker_index)

    def test_action_feature_pass(self):
        """验证 pass 动作的 action feature 结构。"""
        feature, hand = self._get_feature([], extra_cards=[Card("D", "3")])
        self.assertEqual(feature.size, 139)
        self.assertEqual(int(feature[:52].sum()), 0)
        self._assert_one_hot(feature[104:113], 0)
        self._assert_one_hot(feature[113:126], None)
        self._assert_one_hot(feature[126:139], None)
        idx = card_to_id(hand[0])
        self.assertEqual(int(feature[52 + idx]), 1)

    def test_same_rank_by_suit(self):
        """同点数时按花色比较：黑桃 > 红桃 > 梅花 > 方块。"""
        # 单张
        spade_ace = make_action([Card("S", "A")])
        heart_ace = make_action([Card("H", "A")])
        club_ace = make_action([Card("C", "A")])
        diamond_ace = make_action([Card("D", "A")])
        self.assertTrue(can_beat(spade_ace, heart_ace, northern_rule=True))
        self.assertTrue(can_beat(heart_ace, club_ace, northern_rule=True))
        self.assertTrue(can_beat(club_ace, diamond_ace, northern_rule=True))
        self.assertFalse(can_beat(diamond_ace, spade_ace, northern_rule=True))

        # 对子：最大花色决定大小
        pair_sh = make_action([Card("H", "K"), Card("S", "K")])  # 最大黑桃
        pair_dc = make_action([Card("D", "K"), Card("C", "K")])  # 最大梅花
        self.assertTrue(can_beat(pair_sh, pair_dc, northern_rule=True))
        self.assertFalse(can_beat(pair_dc, pair_sh, northern_rule=True))

        # 三张：最大花色决定大小
        triple_dch = make_action(
            [Card("D", "Q"), Card("C", "Q"), Card("H", "Q")]
        )  # 最大红桃
        triple_dcs = make_action(
            [Card("D", "Q"), Card("C", "Q"), Card("S", "Q")]
        )  # 最大黑桃
        self.assertTrue(can_beat(triple_dcs, triple_dch, northern_rule=True))
        self.assertFalse(can_beat(triple_dch, triple_dcs, northern_rule=True))

    def test_straight_edge_cases(self):
        """顺子边界规则：不能含2，A只能当高位，不能循环。"""
        # 含2的顺子非法
        self.assertIsNone(
            make_action(
                [
                    Card("D", "T"),
                    Card("C", "J"),
                    Card("H", "Q"),
                    Card("S", "K"),
                    Card("D", "2"),
                ]
            )
        )
        # A当低位的顺子非法（A2345）
        self.assertIsNone(
            make_action(
                [
                    Card("D", "A"),
                    Card("C", "2"),
                    Card("H", "3"),
                    Card("S", "4"),
                    Card("D", "5"),
                ]
            )
        )
        # 循环顺子非法（QKA23）
        self.assertIsNone(
            make_action(
                [
                    Card("D", "Q"),
                    Card("C", "K"),
                    Card("H", "A"),
                    Card("S", "2"),
                    Card("D", "3"),
                ]
            )
        )
        # 合法的最大顺子 10JQKA
        straight_10jqka = make_action(
            [
                Card("D", "T"),
                Card("C", "J"),
                Card("H", "Q"),
                Card("S", "K"),
                Card("D", "A"),
            ]
        )
        self.assertIsNotNone(straight_10jqka)
        self.assertEqual(straight_10jqka.action_type, "straight")
        # 合法的最小顺子 34567
        straight_34567 = make_action(
            [
                Card("D", "3"),
                Card("C", "4"),
                Card("H", "5"),
                Card("S", "6"),
                Card("D", "7"),
            ]
        )
        self.assertIsNotNone(straight_34567)
        self.assertEqual(straight_34567.action_type, "straight")
        # 最大顺子 > 最小顺子
        self.assertTrue(can_beat(straight_10jqka, straight_34567, northern_rule=True))
        self.assertFalse(can_beat(straight_34567, straight_10jqka, northern_rule=True))

    def test_first_trick_must_contain_d3(self):
        """第一手牌必须包含方块3。"""
        hand = [
            START_CARD,
            Card("S", "A"),
            Card("H", "K"),
            Card("C", "Q"),
            Card("D", "4"),
            Card("S", "5"),
        ]
        actions = get_legal_actions(hand, last_action=None, must_contain_card=True)
        self.assertTrue(actions)
        for action in actions:
            self.assertIn(START_CARD, action.cards)

        actions_no_rule = get_legal_actions(
            hand, last_action=None, must_contain_card=False
        )
        self.assertTrue(
            any(START_CARD not in action.cards for action in actions_no_rule)
        )

    def test_must_play_if_can_beat(self):
        """北方规则：单张/对子/三张严格同类型，有能压的时必须出且不能pass。"""
        last_action = make_action([Card("D", "4"), Card("S", "4")])
        hand = [
            Card("D", "5"),
            Card("S", "5"),
            Card("H", "7"),
        ]
        actions = get_legal_actions(hand, last_action, must_contain_card=False)
        self.assertTrue(actions)
        self.assertTrue(all(action.action_type != "pass" for action in actions))

    def test_no_cross_type_beating(self):
        """非五张牌型之间不能跨牌型压制（如对子不能压顺子）。"""
        last_action = make_action([Card("D", "4"), Card("S", "4")])
        hand = [
            Card("D", "5"),
            Card("C", "6"),
            Card("H", "7"),
            Card("S", "8"),
            Card("D", "9"),
        ]
        actions = get_legal_actions(hand, last_action, must_contain_card=False)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0], PASS_ACTION)

    def test_northern_rule_five_card_as_same_type(self):
        """北方规则：五张牌型视为同类型，可互相压制，有能压的五张牌时必须出（不能pass）。"""
        last_action = make_action(
            [
                Card("D", "4"),
                Card("C", "5"),
                Card("H", "6"),
                Card("S", "7"),
                Card("D", "8"),
            ]
        )
        hand = [
            Card("D", "5"),
            Card("C", "6"),
            Card("H", "7"),
            Card("S", "8"),
            Card("D", "9"),  # 顺子
            Card("D", "T"),
            Card("S", "T"),
            Card("H", "T"),
            Card("C", "T"),
            Card("D", "3"),  # 铁支
        ]
        actions = get_legal_actions(
            hand, last_action, must_contain_card=False, northern_rule=True
        )
        action_types = {a.action_type for a in actions if a != PASS_ACTION}
        self.assertIn("straight", action_types)
        self.assertIn("four_of_a_kind", action_types)
        # 有五张牌能压时必须出，不能pass
        self.assertNotIn(PASS_ACTION, actions)

    def test_northern_rule_cross_type_when_no_same_type(self):
        """北方规则：没有严格同类型的五张牌可压时，允许用其他五张牌型跨牌型压牌（仍不能pass）。"""
        last_action = make_action(
            [
                Card("D", "4"),
                Card("C", "5"),
                Card("H", "6"),
                Card("S", "7"),
                Card("D", "8"),
            ]
        )
        hand = [
            Card("H", "3"),
            Card("H", "5"),
            Card("H", "7"),
            Card("H", "9"),
            Card("H", "J"),  # 同花
            Card("D", "T"),
            Card("S", "T"),
            Card("H", "T"),
            Card("C", "T"),
            Card("D", "3"),  # 铁支
        ]
        actions = get_legal_actions(
            hand, last_action, must_contain_card=False, northern_rule=True
        )
        action_types = {a.action_type for a in actions if a != PASS_ACTION}
        self.assertIn("flush", action_types)
        self.assertIn("four_of_a_kind", action_types)
        # 五张牌视为同类型，有能压的时必须出，不能pass
        self.assertNotIn(PASS_ACTION, actions)

    def test_southern_rule_no_cross_type_for_normal_five_card(self):
        """南方规则：普通五张牌型不允许跨牌型互压（同花顺和铁支除外）。"""
        last_action = make_action(
            [
                Card("D", "4"),
                Card("C", "5"),
                Card("H", "6"),
                Card("S", "7"),
                Card("D", "8"),
            ]
        )
        hand = [
            Card("H", "3"),
            Card("H", "5"),
            Card("H", "7"),
            Card("H", "9"),
            Card("H", "J"),
        ]
        actions = get_legal_actions(
            hand, last_action, must_contain_card=False, northern_rule=False
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0], PASS_ACTION)

    def test_full_game_flow(self):
        """完整环境对局流程：随机策略运行到结束，验证轮次流转和pass规则。"""
        env = rlcard.make("chudadi")
        state, player_id = env.reset()
        total_steps = 0
        max_steps = 1000
        last_leader = None
        consecutive_passes = 0

        while not env.is_over() and total_steps < max_steps:
            legal_action_ids = list(state["legal_actions"].keys())
            action_id = int(env.np_random.choice(legal_action_ids))
            raw_action = state["raw_legal_actions"][legal_action_ids.index(action_id)]
            prev_action = env.game.round.last_action
            prev_current = env.game.round.current_player

            state, next_player_id = env.step(action_id)
            total_steps += 1

            # 验证动作后当前玩家正确推进
            if raw_action == "pass" or raw_action == []:
                # pass 后本轮不可再出：下一家接棒
                self.assertEqual(next_player_id, (prev_current + 1) % 4)
                if prev_action is not None:
                    consecutive_passes += 1
            else:
                # 出牌后成为当前轮的leader
                last_leader = prev_current
                consecutive_passes = 0

            # 验证：如果连续3家pass，第4家（最后出牌者）重新领出
            if consecutive_passes >= 3 and last_leader is not None:
                self.assertEqual(env.game.round.last_action, None)
                self.assertEqual(env.game.round.current_player, last_leader)
                consecutive_passes = 0
                last_leader = None

        self.assertTrue(env.is_over())
        self.assertIsNotNone(env.game.winner_id)
        payoffs = env.get_payoffs()
        self.assertEqual(len(payoffs), 4)
        # 只有胜者得分为正（或零），其余为负（或零）
        self.assertGreater(payoffs[env.game.winner_id], 0)

    def test_pass_then_new_round(self):
        """验证pass后本轮不可再出，且最后出牌者重新领出。"""
        env = rlcard.make("chudadi")
        state, _ = env.reset()

        trace = []

        while not env.is_over():
            legal_action_ids = list(state["legal_actions"].keys())
            action_id = int(env.np_random.choice(legal_action_ids))
            raw_action = state["raw_legal_actions"][legal_action_ids.index(action_id)]
            trace.append((env.game.round.current_player, raw_action))
            state, _ = env.step(action_id)

            # 检查 trace：找到最近一轮的所有动作
            # 如果当前 last_action 为 None，说明新一轮开始，上一轮最后非pass动作的玩家应成为 leader
            if env.game.round.last_action is None and len(trace) >= 2:
                # 找到上一轮最后出牌的玩家
                last_round_actions = list(trace)
                last_non_pass = None
                for pid, act in reversed(last_round_actions[:-1]):
                    if act != "pass" and act != []:
                        last_non_pass = pid
                        break
                if last_non_pass is not None:
                    self.assertEqual(env.game.round.current_player, last_non_pass)
                    # 重置trace以跟踪新一轮
                    trace = []

        self.assertTrue(env.is_over())

    def test_scoring_northern_rule(self):
        """北方规则计分：胜者得其余玩家扣分之和，13张*3，10-12张*2，其余正常。"""
        players = [ChuDaDiPlayer(pid) for pid in range(4)]
        players[0].set_current_hand([])  # winner
        players[1].set_current_hand(
            self._make_cards(
                ["2", "2", "A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4"]
            )  # 13 cards -> 13 * 3 = 39
        )
        players[2].set_current_hand(
            self._make_cards(["2", "A", "K", "Q", "J", "T", "9", "8"])  # 8 cards -> 8
        )
        players[3].set_current_hand(self._make_cards(["A", "K", "Q"]))  # 3 cards -> 3

        # Mark players as having played cards (not "never played")
        for i in range(1, 4):
            players[i].played_cards = [Card("D", "3")]

        payoffs = ChuDaDiJudger(None, northern_rule=True).judge_payoffs(
            players, winner_id=0
        )
        self.assertEqual(payoffs[0], 50)  # winner gets sum of others' losses: 39+8+3=50
        self.assertEqual(payoffs[1], -39)  # 13 cards * 3
        self.assertEqual(payoffs[2], -8)  # 8 cards
        self.assertEqual(payoffs[3], -3)  # 3 cards


if __name__ == "__main__":
    unittest.main()
