import json
import re
from pprint import pprint

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import chess
import chess.engine

from data_processing.mistral_chat import MistralChat, sys_chess, prompt_chess, sys_error, prompt_template
from config import MISTRAL_API_KEY, STOCKFISH_PATH

app = Flask(__name__)
CORS(app)


@app.route('/api/move', methods=['POST'])
def get_computer_move():
    data = request.get_json()
    fen = data['fen']
    engine_name = data['engine']
    pprint(f'Engine name: {engine_name}')

    if engine_name == 'stockfish':
        move_json = get_stockfish_move(fen)
    elif engine_name == 'mistral-7b':
        move_json = get_mistral_move(fen, model='open-mistral-7b')
    elif engine_name == 'mixtral-8x7b':
        move_json = get_mistral_move(fen, model='open-mixtral-8x7b')
    else:
        # Add error logging here:
        logging.error(f'Invalid engine name: {engine_name}')
        move_json = None

    return move_json


def get_stockfish_move(fen):
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    board = chess.Board(fen)

    move = engine.play(board, chess.engine.Limit(time=0.1)).move
    commentary = f"Stockfish plays {move}"
    return jsonify({'move': str(move), 'commentary': commentary})


def get_mistral_move(fen, model):
    board = chess.Board(fen)
    sys = sys_chess()
    prompt = prompt_chess(board)

    llm = MistralChat(MISTRAL_API_KEY)
    llm.add_message(sys)
    llm.add_message(prompt)
    print('****** CONVERSATION HISTORY ******\n ')
    pprint(llm.get_messages()[-1])
    retries, MAX_RETRIES = 0, 15
    while retries < MAX_RETRIES:
        output: str = llm.generate_text(model_name=model)
        print('****** RAW OUTPUT ********\n ')
        pprint(output['content'])
        response_json, error_message = validate_move(output, board)
        llm.add_message(output)
        # print(response_json, error_message)
        if response_json:
            return jsonify(response_json)
        else:
            print('****** ERROR MESSAGE ******\n ')
            pprint(error_message)
            llm.add_message(sys_error())
            llm.add_message(prompt_template("user", error_message))
            llm.add_message(sys)
            llm.add_message(prompt)
            # pprint(llm.get_messages())
            retries += 1

    return jsonify({'error': 'Exceeded maximum retries'})


def validate_move(output: dict, board: chess.Board):
    """Validate the move suggested by the Mistral API."""
    # pprint(output)

    def extract_json(text):
        """finds opening {, closing }, and returns the JSON object in between."""
        start = text.find('{')
        end = text.rfind('}')
        """handles case if {} not found"""
        if start == -1 or end == -1:
            return text
        else:
            return text[start:end + 1]

    # Move legality for validation
    legal_moves: list[chess.Move] = list(board.legal_moves)
    legal_moves_san: list[str] = [board.san(move) for move in legal_moves]

    # Extract JSON object from Mistral API response
    output_text = output['content']
    extracted_json_str: str = extract_json(output_text)
    print("****** EXTRACTED JSON STRING ******\n")
    pprint(extracted_json_str)

    try:
        response_json: dict = json.loads(extracted_json_str)
        # pprint(response_json)
        move: str = response_json['move']
        if move in legal_moves_san:
            return response_json, None
    except (json.JSONDecodeError, KeyError):
        # pprint(output)
        return None, 'Invalid JSON response. Your last response was not a valid JSON object. Please provide your thoughts and move in the correct JSON format: {"thoughts": "Your thoughts here", "move": "Your move in SAN notation"}.'

    try:
        # If not in SAN format, convert from UCI format
        uci_move: str = re.sub(r'[-+#NQRB\s]', '', move)
        uci_move: chess.Move = chess.Move.from_uci(uci_move)
        if uci_move in legal_moves:
            response_json['move'] = board.san(uci_move)
            return response_json, None

    except ValueError:
        return None, f"Invalid move format: '{move}'. Please provide a single move only, using the SAN format. Legal moves in SAN: {', '.join(legal_moves_san)}. Re-prompting..."

    return None, f"Illegal move provided: '{move}'. Please try again, but select from the set of legal moves: {', '.join(legal_moves_san)}."


if __name__ == '__main__':
    app.run(debug=True)
