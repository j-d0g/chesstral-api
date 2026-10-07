from pathlib import Path
from typing import Optional

from .catalog import NANOGPT_MODELS


REPO_ROOT = Path(__file__).resolve().parents[2]


def checkpoint_path(model: str) -> Optional[Path]:
    candidates = (
        REPO_ROOT / "models" / "checkpoints" / model / "ckpt.pt",
        REPO_ROOT / "models" / f"{model}-ckpt.pt",
    )
    return next((path for path in candidates if path.is_file()), None)


def available_models() -> list[str]:
    return [model for model in NANOGPT_MODELS if checkpoint_path(model) is not None]
