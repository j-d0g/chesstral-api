import json


def increment_reprompt(error_message, reprompt_counter):
    if "Illegal move" in error_message:
        reprompt_counter["illegal_move"] += 1
    elif "Invalid move format" in error_message:
        reprompt_counter["invalid_move_format"] += 1
    elif "Invalid JSON response" in error_message:
        reprompt_counter["invalid_json_format"] += 1


def dump_benchmarks(prompt: str, response_json: dict, feature_flag: str, fen: str, pgn_moves: list,
                    reprompt_counts: dict, conversation: list) -> dict:
    benchmarks = {
        "input_features": {
            "pgn": 'p' in feature_flag,
            "fen": 'f' in feature_flag,
            "board": 'b' in feature_flag,
            "legal": 'l' in feature_flag,
            "threats": 't' in feature_flag,
            "prompt": prompt
        },
        "completion": response_json,
        "conversation": conversation,
        "reprompts": reprompt_counts,
        "board_info": {
            "move_num": len(pgn_moves),
            "fen": fen,
            "pgn": pgn_moves,
        }
    }

    with open("self_play_data.json", "a") as file:
        json.dump(benchmarks, file)
        file.write("\n")

    return benchmarks
