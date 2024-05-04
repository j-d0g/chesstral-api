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
        "feature_flags": feature_flag,
        "completion": response_json,
        "context": conversation,
        "reprompt_counter": reprompt_counts,
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
