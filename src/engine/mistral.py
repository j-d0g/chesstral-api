import requests
from engine.base_llm import BaseLLM


class Mistral(BaseLLM):
    def __init__(self):
        super().__init__("MISTRAL_API_KEY")
        self.api_url = "https://api.mistral.ai/v1/"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def get_models(self):
        response = requests.get(self.api_url + "models", headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

    def generate_response(self, max_tokens, temperature, top_p, model_name="open-mistral-7b") -> str:
        data = {
            "model": model_name,
            "messages": self.get_messages(),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
            "safe_prompt": False,
            "random_seed": 1337,
            "response_format": {"type": "json_object"}
        }
        response = requests.post(self.api_url + "chat/completions", headers=self.headers, json=data)

        if response.status_code == 200:
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]
            return generated_text
        else:
            error_msg = f"Error: {response.status_code} - {response.text}"
            return error_msg

    @classmethod
    def prompt_template(cls, role: str, message: str) -> dict[str, str]:
        return {
            "role": role,
            "content": f"[INST] {message} [/INST]"
        }
