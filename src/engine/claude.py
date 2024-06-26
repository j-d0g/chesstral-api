import anthropic
from engine.base_llm import BaseLLM


class Claude(BaseLLM):
    def __init__(self):
        super().__init__("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def get_models(self):
        return {"opus": 'claude-3-opus-20240229', "sonnet": 'claude-3-sonnet-20240229',
                "haiku": 'claude-3-haiku-20240307'}

    def generate_response(self, model_name="claude-3-haiku-20240307", max_tokens=1024, temperature=0.9,
                          top_p=0.95) -> str:
        response = anthropic.Anthropic().messages.create(
            model=model_name,
            system=self.get_messages()[0]['content'],
            messages=self.get_messages()[1:],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=False,
        )
        return response.choices[0].message.content

    @classmethod
    def prompt_template(cls, role: str, message: str) -> dict[str, str]:
        return {
            "role": role,
            "content": message
        }
