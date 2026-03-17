# python3 play_game.py --env chudadi --model_path experiments/dmc_result/chudadi/.pth --unsafe_torch_load
import argparse
import os
import time

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import get_device, set_seed


def _normalize_rank(rank):
    if rank == "T":
        return "10"
    return str(rank)


def _suit_to_symbol(suit):
    suits = {
        "S": "♠",
        "H": "♥",
        "D": "♦",
        "C": "♣",
        "s": "♠",
        "h": "♥",
        "d": "♦",
        "c": "♣",
    }
    return suits.get(suit, str(suit))


def _split_card_str(card_str):
    if not card_str:
        return None, None
    if card_str[0] in ("S", "H", "D", "C", "s", "h", "d", "c"):
        return card_str[0], card_str[1:]
    if card_str[-1] in ("S", "H", "D", "C", "s", "h", "d", "c"):
        return card_str[-1], card_str[:-1]
    return None, None


def _card_to_text(card):
    if card is None:
        return "None"
    if hasattr(card, "suit") and hasattr(card, "rank"):
        suit = _suit_to_symbol(card.suit)
        rank = _normalize_rank(card.rank)
        return f"{suit}{rank}"
    if isinstance(card, str) and len(card) >= 2:
        suit_char, rank_str = _split_card_str(card)
        if suit_char is not None:
            suit = _suit_to_symbol(suit_char)
            rank = _normalize_rank(rank_str)
            return f"{suit}{rank}"
    return str(card)


def _format_cards_text(cards):
    if cards is None:
        return "pass"
    if isinstance(cards, str):
        text = cards.strip()
        if not text or text.lower() == "pass":
            return "pass"
        parts = text.split()
        if len(parts) > 1:
            return " ".join(_card_to_text(part) for part in parts)
        cards = [text]
    if not cards:
        return "pass"
    return " ".join(_card_to_text(card) for card in cards)


def _print_hand(cards, title):
    print(f"{title} {_format_cards_text(cards)}")


def _format_action_text(action):
    if isinstance(action, (list, tuple)):
        return _format_cards_text(action)
    if isinstance(action, str):
        return _format_cards_text(action)
    return str(action)


def _print_legal_actions(raw_actions, limit):
    print(f"Legal actions: {len(raw_actions)}")
    for i, action in enumerate(raw_actions[:limit]):
        print(f"  {i}: {_format_action_text(action)}")
    if len(raw_actions) > limit:
        print("  ...")


def play_game(env, agents, max_show_actions, delay, pause, show_all_hands):
    state, player_id = env.reset()
    step = 0
    while not env.is_over():
        raw = state["raw_obs"]
        print(f"\n{step} | Player{player_id}")
        if show_all_hands:
            info = env.get_perfect_information()
            for pid, hand in enumerate(info["hand_cards"]):
                print(f"P{pid} hand: {_format_cards_text(hand)}")
        else:
            _print_hand(raw["current_hand"], "Current hand:")

        if raw["last_action"]:
            _print_hand(raw["last_action"], "Last action:")
        else:
            print("Last action: None")

        print(f'Cards left: {raw["num_cards_left"]}')
        # _print_legal_actions(state['raw_legal_actions'], max_show_actions) ## 输出可能的牌型

        action = agents[player_id].step(state)
        legal_ids = list(state["legal_actions"].keys())
        action_index = legal_ids.index(action)
        action_raw = state["raw_legal_actions"][action_index]
        print(f"Player{player_id} plays: {_format_action_text(action_raw)}")

        state, player_id = env.step(action)
        step += 1

        if pause:
            input("Press Enter to continue...")
        elif delay > 0:
            time.sleep(delay)

    payoffs = env.get_payoffs()
    print("\nGame over")
    print(f"Payoffs: {payoffs}")


def load_model(model_path, env, device, unsafe_torch_load=False):
    if model_path == "random":
        return RandomAgent(num_actions=env.num_actions)
    if os.path.isfile(model_path):
        import torch

        load_kwargs = {"map_location": device}
        if "weights_only" in torch.load.__code__.co_varnames:
            load_kwargs["weights_only"] = False if unsafe_torch_load else True
        try:
            agent = torch.load(model_path, **load_kwargs)
        except Exception as exc:
            if not unsafe_torch_load and load_kwargs.get("weights_only"):
                raise RuntimeError(
                    "Torch refused to load this checkpoint with weights_only=True. "
                    "If you trust the file, rerun with --unsafe_torch_load."
                ) from exc
            raise
        if hasattr(agent, "set_device"):
            agent.set_device(device)
        return agent
    if os.path.isdir(model_path):
        from rlcard.agents import CFRAgent

        agent = CFRAgent(env, model_path)
        agent.load()
        return agent
    raise ValueError(f"Unknown model path: {model_path}")


def main():
    parser = argparse.ArgumentParser("Play a simple visualized game")
    parser.add_argument("--env", type=str, default="chudadi")
    parser.add_argument("--num_games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max_show_actions", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--pause", action="store_true")
    parser.add_argument("--show_all_hands", action="store_true")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--model_position", type=int, default=0)
    parser.add_argument("--unsafe_torch_load", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    env = rlcard.make(args.env, config={"seed": args.seed})
    agents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)]
    if args.model_path:
        if not (0 <= args.model_position < env.num_players):
            raise ValueError(f"Invalid model_position: {args.model_position}")
        device = get_device()
        agents[args.model_position] = load_model(
            args.model_path,
            env,
            device,
            unsafe_torch_load=args.unsafe_torch_load,
        )
    env.set_agents(agents)

    for game_index in range(args.num_games):
        print(f"\n===== Game {game_index + 1} =====")
        play_game(
            env,
            agents,
            args.max_show_actions,
            args.delay,
            args.pause,
            args.show_all_hands,
        )


if __name__ == "__main__":
    main()

# python3 play_game.py --env chudadi --model_path experiments/dmc_result/chudadi/.pth --unsafe_torch_load
