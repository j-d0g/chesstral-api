import requests


def prompt_template(role: str, message: str) -> dict[str, str]:
    return {
        "role": role,
        "content": f"[INST] {message} [/INST]"
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

    def reset_messages(self):
        self.messages = []

    def generate_text(self, model_name="open-mistral-7b", max_tokens=1024, temperature=0.9):
        data = {
            "model": model_name,
            "messages": self.get_messages(),
            "temperature": temperature,
            "top_p": 0.7,
            "max_tokens": max_tokens,
            "stream": False,
            "safe_prompt": False,
            "random_seed": 1337,
            "response_format": {"type": "json_object"}
        }
        response = requests.post(self.api_url + "chat/completions", headers=self.headers, json=data)

        if response.status_code == 200:
            result = response.json()
            generated_text = result["choices"][0]["message"]
            return generated_text
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
