from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

from repository.chatgpt import ChatGPT
from repository.mistral import Mistral
from service.stockfish_service import get_stockfish_move
from service.llm_service import get_llm_move, extract_san

from dotenv import load_dotenv
import os

app = Flask(__name__)
CORS(app)


class ChessLLMResource:
    def __init__(self):
        self.pgn_moves = []

    def get_computer_move(self):
        data = request.get_json()
        fen, engine_name, last_move = data['fen'], data['engine'], data['lastMove']
        self.pgn_moves.append(extract_san(fen, last_move))

        load_dotenv()

        if 'stockfish' in engine_name:
            return get_stockfish_move(fen, os.getenv('STOCKFISH_PATH'))
        elif 'gpt' in engine_name:
            llm = ChatGPT(os.getenv('GPT_API_KEY'))
        elif 'mistral' or 'mixtral' in engine_name:
            llm = Mistral(os.getenv('MISTRAL_API_KEY'))
        else:
            logging.error(f'Engine not yet supported: {engine_name}')
            return None

        # add pgn_moves list to response before returning

        return jsonify(get_llm_move(self.pgn_moves, fen, llm, engine_name))

    def reset_game(self):
        self.pgn_moves = []
        return jsonify({'message': 'Game reset!'})


chess_llm = ChessLLMResource()


@app.route('/api/move', methods=['POST'])
def get_computer_move():
    return chess_llm.get_computer_move()


@app.route('/api/reset', methods=['POST'])
def reset_game():
    return chess_llm.reset_game()


if __name__ == '__main__':
    app.run(debug=True)
