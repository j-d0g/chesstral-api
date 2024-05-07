import json


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

    with open("self_play_data.json", "a") as file:
        json.dump(data, file)
        file.write("\n")

    return data


def dump_human_eval():
    pass


def dump_game():
    pass
