from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

from repository.chatgpt import ChatGPT
from repository.claude import Claude
from repository.llama import Llama
from repository.mistral import Mistral
from chess_util.prompt_generator import system_chess_prompt
from chess_util.stockfish_engine import get_stockfish_move
from service.llm_service import get_llm_move

from dotenv import load_dotenv
import os

app = Flask(__name__)
CORS(app)


@app.route('/api/move', methods=['POST'])
def get_computer_move():
    data = request.get_json()
    fen, engine_name, moves, context = data['fen'], data['engine'], data['pgn'], data['context']
    load_dotenv()

    if 'stockfish' in engine_name:
        return get_stockfish_move(fen, os.getenv('STOCKFISH_PATH'))
    if 'gpt' in engine_name:
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

    llm.add_message("system", system_chess_prompt())
    llm.add_messages(context[1:])
    response = get_llm_move(moves, fen, llm, engine_name, 'p')

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
