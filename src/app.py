from http import HTTPStatus
from pprint import pprint

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

from engine.chatgpt import ChatGPT
from engine.claude import Claude
from engine.llama import Llama
from engine.mistral import Mistral
from engine.stockfish import Stockfish
from service import move_service

app = Flask(__name__)
CORS(app)


@app.route('/api/move', methods=['POST'])
def get_computer_move():
    data = request.get_json()

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
        return None

    response = move_service.get_llm_move(moves, fen, llm, engine_name, context, 'p')

    return jsonify(response)


@app.route('/api/rate_move', methods=['POST'])
def rate_move():
    data = request.get_json()
    if data:
        move_service.rate_move(data)
    else:
        logging.error('No data received for rating move.')
    return "Move rated successfully", HTTPStatus.OK


@app.route('/api/eval', methods=['POST'])
def eval_board():
    data = request.get_json()
    fen = data['fen']
    stockfish = Stockfish()
    score = stockfish.get_stockfish_evaluation(fen)
    return jsonify(score)


if __name__ == '__main__':
    app.run(debug=True)
