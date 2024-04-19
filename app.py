from pprint import pprint

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

from repository.chatgpt import ChatGPT
from repository.llama import Llama
from repository.mistral import Mistral
from service.stockfish_service import get_stockfish_move
from service.llm_service import get_llm_move

from dotenv import load_dotenv
import os


app = Flask(__name__)
CORS(app)


@app.route('/api/move', methods=['POST'])
def get_computer_move():
    data = request.get_json()
    fen = data['fen']
    engine_name = data['engine']
    load_dotenv()

    pprint(f'Engine name: {engine_name}')

    if 'stockfish' in engine_name:
        return get_stockfish_move(fen, os.getenv('STOCKFISH_PATH'))
    elif 'gpt' in engine_name:
        llm = ChatGPT(os.getenv('GPT_API_KEY'))
    elif 'meta' in engine_name:
        llm = Llama(os.getenv('REPLICATE_API_KEY'))
    elif 'mistral' or 'mixtral' in engine_name:
        llm = Mistral(os.getenv('MISTRAL_API_KEY'))
    else:
        logging.error(f'Engine not yet supported: {engine_name}')
        return None

    return jsonify(get_llm_move(fen, llm, engine_name))


if __name__ == '__main__':
    app.run(debug=True)
