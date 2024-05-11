import json

import pandas as pd


def dump_data(response_json: dict, feature_flag: str, fen: str, pgn_moves: list,
              reprompt_counts: dict, conversation: list) -> dict:
    data = {
        "prompt": {
            "feature_flags": feature_flag,
            "context": conversation,
            "completion": response_json,
            "reprompt_counter": reprompt_counts,
        },
        "board": {
            "move_num": len(pgn_moves),
            "fen": fen,
            "pgn": pgn_moves,
        }
    }

    with open("src/data/game_data/self_play_data.json", "a") as file:
        json.dump(data, file)
        file.write("\n")

    return data


def read_data():
    games = pd.read_json('src/data/game_data/self_play_data.json', lines=True)
    df1 = pd.json_normalize(games['prompt'])
    df2 = pd.json_normalize(games['board'])
    games = pd.concat([df1, df2], axis=1)

    return games


def dump_human_eval(ratings: dict):
    with open("src/data/game_data/human_eval_data.json", "a") as file:
        json.dump(ratings, file)
        file.write("\n")


def dump_game():
    pass
