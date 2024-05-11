from collections import defaultdict

import chess

import repository.benchmarks
from engine.base_llm import BaseLLM
from util.chess_translations import user_chess_prompt, system_chess_prompt
from repository.benchmarks import dump_data
from service.move_validation import validate_json, validate_move, increment_reprompt


def get_llm_move(pgn_moves: list[str], fen: str, llm: BaseLLM, model_name: str, context: list, feature_flags: str,
                 max_retries: int = 10, reset_cycle: int = 5) -> dict:
    """
    Generates a chess move using the Mistral API.
    :param pgn_moves: list of moves in SAN string format
    :param fen: string representing the current board state
    :param feature_flags: list of booleans corresponding to what features to include in the prompt
    :param llm: selected llm chess-engine instance
    :param model_name: Name of the language model to use (optional)
    :param max_retries: Maximum number of retries before giving up (default: 15)
    :param reset_cycle: number of retries before resetting the error messages
    :return: JSON object containing the move and thoughts, or an error message
    """

    # Initialise board using FEN, and push the user's last move
    board = chess.Board(fen)
    if board.is_checkmate():
        return {"thoughts": 'Checkmate!', "move": '#'}

    # Initialise Prompt-Counter
    reprompt_counter = defaultdict(int)

    # Initialise Prompt and Context
    prompt: str = user_chess_prompt(board, pgn_moves, feature_flags)
    llm.add_message("system", system_chess_prompt())
    llm.add_messages(context[1:])

    # Reprompt Loop
    for retry in range(max_retries):
        if (retry + 1) % reset_cycle == 0:
            [llm.pop_message() for _ in range(retry % reset_cycle)]

        output: str = llm.grab_text(prompt, model_name=model_name)
        print(llm.get_messages())
        response_json: dict = validate_json(output)

        if 'reprompt' not in response_json:
            response_json: dict = validate_move(board, response_json)

        if 'reprompt' not in response_json:
            benchmarks: dict = dump_data(
                response_json,
                feature_flags,
                fen,
                pgn_moves,
                reprompt_counter,
                llm.get_messages()
            )
            [llm.pop_message() for _ in range(retry)]
            return benchmarks

        prompt = f"{response_json['reprompt']}. Previous prompt: '''{user_chess_prompt(board, pgn_moves, feature_flags)}'''"
        increment_reprompt(response_json['reprompt'], reprompt_counter)

    raise ValueError("Max retries exceeded. Unable to generate a valid response.")

def rate_move(data: dict) -> int:
    human_eval_data = {
        "fen": data['fen'],
        "llm": data['engineName'],
        "move": data['move'],
        "pgn": data['moveSequence'],
        "move_quality": data['quality'],
        "correctness": data['correctness'],
        "relevance": data['relevance'],
        "salience": data['salience']
    }

    repository.benchmarks.dump_human_eval(human_eval_data)


def upgrade_model(model_name: str) -> str:
    """
    A solution to consistently bad re-prompts: upgrading to larger models after a certain number of retries.
    :param model_name: current model name
    :return: next model name
    """
    if "mistral" in model_name or "mixtral" in model_name:
        models = {
            "1": "open-mistral-7b",
            "2": "open-mixtral-8x7b",
            "3": "open-mixtral-8x22b",
            "4": "mistral-medium-latest",
            "5": "mistral-large-latest"
        }
    elif "gpt" in model_name:
        models = {
            "1": "gpt-3.5-turbo-0125",
            "2": "gpt-4-turbo"
        }
    else:
        raise ValueError("Model not supported")

    inverted_models = {v: k for k, v in models.items()}
    current_model = inverted_models[model_name]
    next_model = int(current_model) + 1
    if next_model > len(models):
        return model_name

    return models[str(next_model)]