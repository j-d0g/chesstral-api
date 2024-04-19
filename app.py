from pprint import pprint

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

from chess_engine.mistral import MistralChat
from chess_engine.stockfish import get_stockfish_move
from chess_engine.engine import get_mistral_move
from config import MISTRAL_API_KEY

app = Flask(__name__)
CORS(app)


@app.route('/api/move', methods=['POST'])
def get_computer_move():
    data = request.get_json()
    fen = data['fen']
    engine_name = data['engine']
    pprint(f'Engine name: {engine_name}')

    if 'stockfish' in engine_name:
        move_json = get_stockfish_move(fen)
    elif 'mistral' or 'mixtral' in engine_name:
        llm = MistralChat(MISTRAL_API_KEY)
        move_json = get_mistral_move(fen, engine_name, llm)
    else:
        logging.error(f'Invalid engine name: {engine_name}')
        move_json = None

    return jsonify(move_json)


if __name__ == '__main__':
    app.run(debug=True)
