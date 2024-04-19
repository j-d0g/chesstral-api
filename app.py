from pprint import pprint

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

from chess_repository.chatgpt import ChatGPT
from chess_repository.mistral import MistralChat
from chess_service.stockfish import get_stockfish_move
from chess_service.engine import get_llm_move
from config import MISTRAL_API_KEY, GPT_API_KEY

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
    elif 'gpt' in engine_name:
        print("APP->GPT")
        gpt = ChatGPT(GPT_API_KEY)
        move_json = get_llm_move(fen, gpt, engine_name)
    elif 'mistral' or 'mixtral' in engine_name:
        mistral = MistralChat(MISTRAL_API_KEY)
        move_json = get_llm_move(fen, mistral, engine_name)
    else:
        logging.error(f'Invalid engine name: {engine_name}')
        move_json = None

    return jsonify(move_json)


if __name__ == '__main__':
    app.run(debug=True)
