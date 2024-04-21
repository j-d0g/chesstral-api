import json
import re
import chess
from pprint import pprint

from repository.base_llm import BaseLLM
from service.chess_util.prompt_generator import system_chess_prompt, user_chess_prompt


def extract_benchmarks(prompt: str, response_json: dict, feature_flag: str, fen: str, pgn_moves: list, reprompt_counts: dict) -> dict:
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
        "re-prompts": reprompt_counts,
        "board-info": {
            "move_num": len(pgn_moves),
            "fen": fen,
            "pgn": pgn_moves,
        }
    }

    return benchmarks


def persist_benchmarks(benchmarks: dict)-> None:
    """
    Persists the benchmarks to a JSON file.
    :param benchmarks:
    :return: None
    """
    with open("self_play_data.json", "a") as file:
        json.dump(benchmarks, file)
        file.write("\n")


def generate_move(pgn_moves: list[str], fen: str, llm: BaseLLM, model_name: str, feature_flags: str, max_retries: int = 15) -> dict:
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
    # Initialise Prompt-Counter
    reprompt_counter = {
        "illegal_move": 0,
        "invalid_move_format": 0,
        "invalid_json_format": 0,
    }

    # Initialise board using FEN, and push the user's last move
    board = chess.Board(fen)
    board.push_san(pgn_moves[-1])

    # Early Exit if Checkmate
    if board.is_checkmate():
        return {"thoughts": 'Checkmate!', "move": '#'}

    # Generate system and user prompt from templates, add to LLM messages
    sys = llm.prompt_template("system", system_chess_prompt())
    user = llm.prompt_template("user", user_chess_prompt(board, pgn_moves, feature_flags))
    llm.add_messages([sys, user])

    print('****** INPUT ******\n ')
    pprint(llm.get_messages())

    # HANDLE GENERATION & RETRIES
    for retry in range(max_retries):

        # If reached 5 consecutive retries, reset conversation context and start again
        if (retry + 1) % 5 == 0:
            llm.reset_messages()
            llm.add_messages([sys, user])

        # Generate move, thoughts JSON using LLM
        output: str = llm.generate_text(model_name=model_name)
        response_json, error_message = validate_response(output, board)

        # If response is valid, extract and generate benchmark metrics and return response
        if response_json:
            benchmarks = extract_benchmarks(
                user, 
                response_json, 
                feature_flags, 
                fen, 
                pgn_moves, 
                reprompt_counter
            )
            persist_benchmarks(benchmarks)
            return benchmarks

        # Else if error, re-prompt the user with the error-reprompt message
        regenerate_message = f"{error_message}. Previous prompt: '''{user_chess_prompt(board, pgn_moves, feature_flags)}'''"
        llm.add_message(llm.prompt_template("assistant", output))
        llm.add_message(llm.prompt_template("user", regenerate_message))

        # Increment the corresponding re-prompt counter
        if "Illegal move" in error_message:
            reprompt_counter["illegal_move"] += 1
        elif "Invalid move format" in error_message:
            reprompt_counter["invalid_move_format"] += 1
        elif "Invalid JSON response" in error_message:
            reprompt_counter["invalid_json_format"] += 1

    # Return error message if maximum retries exceeded
    return {'error': 'Exceeded maximum retries'}


def validate_response(output: str, board: chess.Board):
    """
    Validates the move generated by the Mistral API.
    :param output: Output string from the Mistral API
    :param board: Current chess board state
    :return: Tuple containing the response JSON (move and thoughts) and an error message (if any)
    """
    legal_moves: list[chess.Move] = list(board.legal_moves)
    cleaned_json: str = clean_json(output)

    # Handle JSON Validation
    try:
        response_json: dict = json.loads(cleaned_json)
        san_move: str = response_json['move']
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return handle_json_error(e, cleaned_json)

    print("****** RESPONSE JSON ******\n")
    pprint(response_json)

    # Handle Move Validation
    try:
        san_move = san_move.replace(' ', '')
        chess_move: chess.Move = board.parse_san(san_move)
        if chess_move in legal_moves:
            # Update response JSON with the cleaned SAN move
            response_json['move'] = san_move
            return response_json, None
        else:
            raise chess.IllegalMoveError
    except Exception as e:
        return handle_move_error(e, san_move, board)


def clean_json(text: str) -> str:
    """
    Cleans the text into a more readable JSON string, removing excess characters that interfere with parsing json/move.
    :param text: Input text string
    :return: Extracted JSON string, or the original text if no JSON is found
    """
    text = re.sub(r'\n', r'', text)
    start = text.find('{')
    end = text.rfind('}')
    return text[start:end + 1] if start != -1 and end != -1 else text


def handle_json_error(error: Exception, json_str: str):
    """
    Returns error-specific prompts to regenerate response for JSON errors.
    :param error: The exception object representing the JSON error
    :param json_str: Response JSON string
    :return: Tuple containing None and the corresponding error message
    """
    if isinstance(error, json.JSONDecodeError):
        return None, f'Invalid JSON response: """{json_str}""" is not a valid JSON object. Regenerate your response, providing your thoughts and move in the correct JSON format: {{"thoughts": "Your reasoning-steps here", "move": "Your move in SAN notation"}}.'
    elif isinstance(error, (KeyError, TypeError)):
        return None, f'Invalid JSON response: """{json_str}""" is missing the "move" key. Regenerate your response, providing the move key in the correct JSON format: {{"thoughts": "Your reasoning-steps here", "move": "Your move in SAN notation"}}.'


def handle_move_error(error, move, board):
    """
    Returns error-specific prompts to regenerate response for chess-move errors.
    :param error: The exception object representing the move error
    :param move: The move that caused the error
    :param board: Current chess board state
    :return: Tuple containing None and the corresponding error message
    """
    legal_moves: list[chess.Move] = list(board.legal_moves)
    legal_moves_san: list[str] = [board.san(chess_move) for chess_move in legal_moves]
    legal_moves_uci: list[str] = [chess_move.uci() for chess_move in legal_moves]

    if isinstance(error, chess.IllegalMoveError):
        return None, f"Illegal move: '{move}'. Regenerate your response to my last prompt, but this time provide a single, legal move in SAN format. Here are the current legal moves in SAN you can make: '''{', '.join(legal_moves_san)}'''."
    elif isinstance(error, chess.AmbiguousMoveError):
        return None, f"Ambiguous move: '{move}'. Regenerate your response to my last prompt, but this time using either long-SAN by specifying the file of the origin piece (i.e Nhg8 instead of Ng8), or UCI format (i.e f6g8). Legal UCI moves: '''{', '.join(legal_moves_uci)}'''."
    elif isinstance(error, chess.InvalidMoveError):
        try:
            # If move was an invalid SAN format, attempt to clean and convert from UCI format
            uci_move: str = re.sub(r'[-+#nqrkNQRBK]', '', move)
            uci_move: chess.Move = chess.Move.from_uci(uci_move)
            if uci_move in legal_moves:
                return {'move': board.san(uci_move)}, None
            else:
                return None, f"Illegal move: '{uci_move}'. Regenerate your response to my last prompt, but this time provide a single, legal move in SAN format. Legal SAN moves: '''{', '.join(legal_moves_san)}'''."
        except (ValueError, chess.InvalidMoveError):
            return None, f"Invalid move format: '{move}'. Regenerate your response to my last prompt, but this time provide a single, legal move using either SAN-format or UCI-format. Legal SAN moves: '''{', '.join(legal_moves_san)}'''."
    else:
        return None, f'Invalid move: {move}. Regenerate your response but choose a valid SAN move: {legal_moves_san}.'


def extract_san(fen, last_move):
    """
    Pushes the last move to the board and extracts the SAN move.

    :param fen:
    :param last_move:
    :return:
    """
    uci_move = last_move['from'] + last_move['to']
    chess_move = chess.Move.from_uci(uci_move)
    board = chess.Board(fen)

    return board.san(chess_move)
