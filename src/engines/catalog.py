from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EngineDefinition:
    name: str
    models: tuple[str, ...]
    description: str
    required_env: Optional[str] = None


NANOGPT_MODELS = (
    "small-8",
    "small-16",
    "small-24",
    "small-36",
    "medium-12",
    "medium-16",
    "large-16",
)

ENGINE_CATALOG = {
    "nanogpt": EngineDefinition(
        name="nanogpt",
        models=NANOGPT_MODELS,
        description="Local chess-trained NanoGPT models",
    ),
    "stockfish": EngineDefinition(
        name="stockfish",
        models=("default",),
        description="Traditional chess engine",
    ),
    "openai": EngineDefinition(
        name="openai",
        models=("o1", "o1-mini", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"),
        description="OpenAI models including o1 reasoning",
        required_env="OPENAI_API_KEY",
    ),
    "anthropic": EngineDefinition(
        name="anthropic",
        models=(
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-opus-4-20250514",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
        ),
        description="Anthropic Claude models",
        required_env="ANTHROPIC_API_KEY",
    ),
    "gemini": EngineDefinition(
        name="gemini",
        models=(
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.5-flash-preview-05-20",
            "gemini-2.5-pro-preview-06-05",
        ),
        description="Google Gemini models",
        required_env="GOOGLE_API_KEY",
    ),
    "deepseek": EngineDefinition(
        name="deepseek",
        models=("deepseek-r1", "deepseek-r1-distill-llama-70b", "deepseek-chat", "deepseek-coder"),
        description="DeepSeek models including R1 reasoning",
        required_env="DEEPSEEK_API_KEY",
    ),
}
