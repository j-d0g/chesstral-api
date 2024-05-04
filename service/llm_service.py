from collections import defaultdict

import chess
from pprint import pprint

from repository.base_llm import BaseLLM
from chess_util.prompt_generator import user_chess_prompt
from service.persister import increment_reprompt, dump_data
from service.validator import validate_json, validate_move


def generate_move(pgn_moves: list[str], fen: str, llm: BaseLLM, model_name: str, feature_flags: str,
                  max_retries: int = 10) -> dict:
    """
    Generates a chess move using the Mistral API.
    :param feature_flags: list of booleans corresponding to what features to include in the prompt
    :param pgn_moves: list of moves in SAN string format
    :param fen: string representing the current board state
    :param llm: Language model instance
    :param model_name: Name of the language model to use (optional)
    :param max_retries: Maximum number of retries before giving up (default: 15)
    :return: JSON object containing the move and thoughts, or an error message
    """
    # Initialise board using FEN, and push the user's last move
    board = chess.Board(fen)
    board.push_san(pgn_moves[-1])
    if board.is_checkmate():
        return {"thoughts": 'Checkmate!', "move": '#'}

    # Initialise Prompt-Counter
    reprompt_counter = defaultdict(int)
    prompt: str = user_chess_prompt(board, pgn_moves, feature_flags)

    # HANDLE GENERATION & RETRIES
    for retry in range(max_retries):
        reset_freq = 5
        if (retry + 1) % reset_freq == 0:
            model_name = upgrade_model(model_name)
            [llm.pop_message() for _ in range(retry%reset_freq)]

        output: str = llm.grab_text(prompt, model_name=model_name)

        print('****** INPUT ******\n ')
        pprint(llm.get_messages())

        response_json: dict = validate_json(output)

        if 'error' not in response_json:
            response_json: dict = validate_move(board, response_json)

        if 'error' not in response_json:
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

        prompt = f"{response_json['error']}. Previous prompt: '''{user_chess_prompt(board, pgn_moves, feature_flags)}'''"
        increment_reprompt(response_json['error'], reprompt_counter)

    raise ValueError("Max retries exceeded. Unable to generate a valid response.")


def upgrade_model(model_name: str) -> str:
    """
    A solution to consistently bad re-prompts: upgrading to larger models after a certain number of retries.
    :param model_name:
    :return:
    """
    if "mistral" in model_name or "mixtral" in model_name:
        models = {"1": "open-mistral-7b", "2": "open-mixtral-8x7b", "3": "open-mixtral-8x22b",
                  "4": "mistral-medium-latest", "5": "mistral-large-latest"}
    elif "gpt" in model_name:
        models = {"1": "gpt-3.5-turbo-0125", "2": "gpt-4-turbo"}
    else:
        raise ValueError("Model not supported")

    return upgrade(models, model_name)


def upgrade(models, model_name):
    inverted_models = {v: k for k, v in models.items()}
    current_model = inverted_models[model_name]
    next_model = int(current_model) + 1
    if next_model > len(models):
        return model_name
    return models[str(next_model)]
