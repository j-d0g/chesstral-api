from abc import ABC, abstractmethod


class BaseLLM(ABC):

    @abstractmethod
    def __init__(self, api_key):
        self.api_key = api_key
        self.messages = []

    @abstractmethod
    def get_models(self):
        pass

    def add_message(self, message) -> None:
        self.messages.append(message)

    def get_messages(self) -> list:
        return self.messages

    def reset_messages(self):
        self.messages = []

    @abstractmethod
    def generate_text(self, model_name: str, max_tokens: int, temperature: float, top_p: float) -> str:
        pass

    @classmethod
    @abstractmethod
    def prompt_template(cls, role: str, message: str):
        pass