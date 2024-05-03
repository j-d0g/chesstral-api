from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

from repository.base_llm import BaseLLM
from repository.chatgpt import ChatGPT
from repository.claude import Claude
from repository.llama import Llama
from repository.mistral import Mistral
from service.chess_util.prompt_generator import system_chess_prompt
from service.stockfish_service import get_stockfish_move
from service.llm_service import generate_move, to_san

from dotenv import load_dotenv
import os

app = Flask(__name__)
CORS(app)


class ChessLLMResource:
    def __init__(self):
        self.pgn_moves = []
        self.conversation = []

    def get_computer_move(self):
        data = request.get_json()
        fen, engine_name, last_move = data['fen'], data['engine'], data['lastMove']
        self.pgn_moves.append(to_san(fen, last_move))
        load_dotenv()

        if 'stockfish' in engine_name:
            return get_stockfish_move(fen, os.getenv('STOCKFISH_PATH'))
        elif 'gpt' in engine_name:
            llm = ChatGPT(api_key=os.getenv('GPT_API_KEY'))
        elif 'mistral' in engine_name or 'mixtral' in engine_name:
            llm = Mistral(api_key=os.getenv('MISTRAL_API_KEY'))
        elif 'llama' in engine_name:
            llm = Llama(api_key=os.getenv('REPLICATE_API_KEY'))
        elif 'claude' in engine_name:
            llm = Claude(api_key=os.getenv('ANTHROPIC_API_KEY'))
        else:
            logging.error(f'Engine not yet supported: {engine_name}')
            return None

        llm.reset_messages()
        llm.add_message("system", system_chess_prompt())
        if data['contextOn']:
            llm.add_messages(self.conversation[1:])
        response = generate_move(self.pgn_moves, fen, llm, engine_name, 'p')
        self.conversation = llm.get_messages()
        self.pgn_moves.append(response['completion']['move'])

        return jsonify(response)

    def reset_game(self):
        self.pgn_moves = []
        self.conversation = []
        return jsonify({'message': 'Game reset!'})

    def reset_context(self):
        self.conversation = []
        return jsonify({'message': 'Context reset!'})


chess_llm = ChessLLMResource()


@app.route('/api/move', methods=['POST'])
def get_computer_move():
    return chess_llm.get_computer_move()


@app.route('/api/reset', methods=['POST'])
def reset_game():
    return chess_llm.reset_game()


@app.route('/api/reset_context', methods=['POST'])
def reset_context():
    return chess_llm.reset_context()


if __name__ == '__main__':
    app.run(debug=True)
