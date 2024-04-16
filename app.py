from pprint import pprint

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

from chess_engine.mistral import get_mistral_move
from chess_engine.stockfish import get_stockfish_move

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
        logging.error(f'Invalid engine name: {engine_name}')
        move_json = None

    return move_json


if __name__ == '__main__':
    app.run(debug=True)
