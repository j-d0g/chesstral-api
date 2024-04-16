from typing import List, Union, Dict

import chess
import requests
from data_processing.prompt_generator import role_to_str, generate_prompt
from pydantic import BaseModel, Field


class ChessMove(BaseModel):
    thoughts: str = Field(description="Board analysis, thoughts or reasoning steps towards the best move.")
    move: str = Field(description="Your move, formatted in Standard Algebraic Notation (SAN).")


def sys_chess() -> dict[str, str]:
    """ Returns a system template for Mistral chat."""

    system_msg = ('[INST] You are an auto-regressive language model that is brilliant at reasoning and playing chess. '
                  'Your goal is to use your reasoning and chess skills to produce the best chess move given a board position. '
                  'Since you are autoregressive, each token you produce is another opportunity to use computation, therefore '
                  'you always spend a few sentences using your knowledge of chess and step-by-step thinking to comment on the '
                  'board-state, tactics, and possible moves before deducing the best possible move. Return a single valid JSON '
                  'object: {"thoughts": "Your thoughts here", "move": "Your move in SAN notation"} [/INST] '
                  )
    return prompt_template("system", system_msg)


def sys_error():
    """ Returns a system error template for Mistral chat."""

    system_msg = (
        '[INST] You are a helpful assistant. Your task is to review the previous interaction, correct any mistakes in your answer, '
        'and provide a new response that adhere\'s to the correct format specified [/INST] '
    )

    return prompt_template("system", system_msg)


def prompt_template(role: str, message: str):
    return {
        "role": role,
        "content": f"[INST] {message} [/INST]"
    }


def prompt_chess(board) -> dict[str, str]:
    """ Returns a content template for Mistral chat."""

    role = role_to_str(board)
    board_str = generate_prompt(board, pgn=False, positions=False, legalmoves=True, threats=False)
    prompt = "\n".join([role, board_str])
    template = f"[INST] {prompt} [/INST]"

    return {
        "role": "user",
        "content": template,
    }


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
