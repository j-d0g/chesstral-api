import json
import os
from datetime import datetime, timezone
from pathlib import Path

from core.types import MoveRatingRequest


def ratings_path() -> Path:
    configured_path = os.getenv("RATINGS_PATH")
    if configured_path:
        return Path(configured_path)
    return Path(__file__).resolve().parents[2] / "data" / "ratings.jsonl"


def save_rating(rating: MoveRatingRequest) -> None:
    path = ratings_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    entry = rating.model_dump()
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as ratings_file:
        ratings_file.write(json.dumps(entry) + "\n")
