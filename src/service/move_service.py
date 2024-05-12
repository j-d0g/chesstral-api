import hashlib
import logging
from collections import defaultdict
from datetime import datetime

import chess

from engine.base_llm import BaseLLM
from engine.chatgpt import ChatGPT
from engine.claude import Claude
from engine.llama import Llama
from engine.mistral import Mistral
from engine.stockfish import Stockfish
from repository import game_repository
from util.chess_translations import user_chess_prompt, system_chess_prompt
from service.move_validation import validate_json, validate_move, increment_reprompt


def get_computer_move(data: dict) -> dict:
    fen, engine_name, moves, context = data['fen'], data['engine'], data['pgn'], data['context']

    if 'stockfish' in engine_name:
        engine = Stockfish()
        return engine.get_stockfish_move(fen)
    if 'gpt' in engine_name:
        llm = ChatGPT()
    elif 'mistral' in engine_name or 'mixtral' in engine_name:
        llm = Mistral()
    elif 'llama' in engine_name:
        llm = Llama()
    elif 'claude' in engine_name:
        llm = Claude()
    else:
        logging.error(f'Engine not yet supported: {engine_name}')
        return {}

    return get_llm_move(moves, fen, llm, engine_name, context, 'p')


def get_llm_move(pgn_moves: list[str], fen: str, llm: BaseLLM, model_name: str, context: list, feature_flags: str,
                 max_retries: int = 10, reset_cycle: int = 5) -> dict:
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
            benchmarks: dict = game_repository.dump_data(
                response_json,
                feature_flags,
                fen,
                pgn_moves,
                reprompt_counter,
                llm.get_messages(),
                generate_move_id(llm.get_messages())
            )
            [llm.pop_message() for _ in range(retry)]
            return benchmarks

        prompt = f"{response_json['reprompt']}. Previous prompt: '''{user_chess_prompt(board, pgn_moves, feature_flags)}'''"
        increment_reprompt(response_json['reprompt'], reprompt_counter)

    raise ValueError("Max retries exceeded. Unable to generate a valid response.")


def get_stockfish_evaluation(data: dict) -> dict:
    fen = data['fen']
    engine = Stockfish()
    return engine.get_stockfish_evaluation(fen)


def save_move_rating(data: dict):
    human_eval_data = {
        "uuid": data['uuid'],
        "fen": data['fen'],
        "llm": data['engineName'],
        "move": data['move'],
        "pgn": data['moveSequence'],
        "move_quality": data['quality'],
        "correctness": data['correctness'],
        "relevance": data['relevance'],
        "salience": data['salience']
    }

    game_repository.dump_move_rating(human_eval_data)


def generate_move_id(conversation: list) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    conversation_str = ''.join(str(message) for message in conversation)
    hash_input = conversation_str + timestamp
    move_id = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
    return move_id


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
