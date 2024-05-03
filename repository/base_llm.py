from abc import ABC, abstractmethod


class BaseLLM(ABC):

    @abstractmethod
    def __init__(self):
        self.messages = []

    @abstractmethod
    def get_models(self):
        pass

    def add_message(self, role: str, message) -> None:
        self.messages.append(self.prompt_template(role, message))

    def add_messages(self, messages: list) -> None:
        self.messages.extend(messages)

    def pop_message(self):
        self.messages = self.messages[:-1]

    def get_messages(self) -> list:
        return self.messages

    def reset_messages(self):
        self.messages = self.messages[:1]

    def grab_text(self, prompt: str, model_name: str, max_tokens: int = 512, temperature: float = 0.9, top_p: float = 0.95):
        self.add_message("user", prompt)

        try:
            response = self.generate_response(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                model_name=model_name,
            )
        except ValueError as e:
            print(str(e))
            return None
        self.add_message("assistant", response)

        return response

    @abstractmethod
    def generate_response(self, max_tokens: int, temperature: float, top_p: float, model_name: str) -> str:
        pass

    @classmethod
    @abstractmethod
    def prompt_template(cls, role: str, message: str):
        pass
