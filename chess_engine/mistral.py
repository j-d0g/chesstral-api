from typing import List, Union, Dict

import chess
import requests
import json
import re
from flask import jsonify

from pprint import pprint
from data_processing.prompt_generator import role_to_str, generate_prompt
from pydantic import BaseModel, Field

from config import MISTRAL_API_KEY


class ChessMove(BaseModel):
    thoughts: str = Field(description="Board analysis, thoughts or reasoning steps towards the best move.")
    move: str = Field(description="Your move, formatted in Standard Algebraic Notation (SAN).")


def prompt_template(role: str, message: str) -> dict[str, str]:
    return {
        "role": role,
        "content": f"[INST] {message} [/INST]"
    }


def sys_chess() -> dict[str, str]:
    """ Returns a system template for Mistral chat."""

    system_msg = ('You are an auto-regressive language model that is brilliant at reasoning and playing chess. '
                  'Your goal is to use your reasoning and chess skills to produce the best chess move given a board position. '
                  'Since you are autoregressive, each token you produce is another opportunity to use computation, therefore '
                  'you always spend a few sentences using your knowledge of chess and step-by-step thinking to comment on the '
                  'board-state, tactics, and possible moves before deducing the best possible move. Return a single valid JSON '
                  'object: {"thoughts": "Your thoughts here", "move": "Your move in SAN notation"}.'
                  )
    return prompt_template("system", system_msg)


def sys_error() -> dict[str, str]:
    """ Returns a system error template for Mistral chat."""

    system_msg = (
        'You are a helpful assistant. Your task is to review the previous interaction, correct any mistakes in your answer, '
        'and provide a new response that adhere\'s to the correct format specified.'
    )

    return prompt_template("system", system_msg)


def user_chess(board: chess.Board) -> dict[str, str]:
    """ Returns a content template for Mistral chat."""

    role = role_to_str(board)
    board_str = generate_prompt(board, pgn=False, positions=False, legalmoves=True, threats=False)
    prompt = "\n".join([role, board_str])
    template = f"[INST] {prompt} [/INST]"

    return {
        "role": "user",
        "content": template,
    }


def get_mistral_move(fen: str, model: str):
    board = chess.Board(fen)
    sys = sys_chess()
    prompt = user_chess(board)

    llm = MistralChat(MISTRAL_API_KEY)
    llm.add_message(sys); llm.add_message(prompt)

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
            retries += 1

    return jsonify({'error': 'Exceeded maximum retries'})


def validate_move(output: dict, board: chess.Board):
    """Validate the move suggested by the Mistral API."""
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
        move: str = response_json['move']
        if move in legal_moves_san:
            return response_json, None
    except (json.JSONDecodeError, KeyError):
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


class MistralChat:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://api.mistral.ai/v1/"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.messages = []

    def get_models(self):
        response = requests.get(self.api_url + "models", headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

    def add_message(self, message):
        self.messages.append(message)

    def get_messages(self):
        return self.messages

    def generate_text(self, model_name="open-mistral-7b", max_tokens=1024, temperature=0.8):
        data = {
            "model": model_name,
            "messages": self.get_messages(),
            "temperature": temperature,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "stream": False,
            "safe_prompt": False,
            "random_seed": 1337,
            "response_format": {"type": "json_object", "schema": ChessMove.model_json_schema()}
        }
        response = requests.post(self.api_url + "chat/completions", headers=self.headers, json=data)

        if response.status_code == 200:
            result = response.json()
            generated_text = result["choices"][0]["message"]
            return generated_text
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
