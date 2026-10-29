import asyncio
import json

import chess
import pytest
from fastapi.testclient import TestClient

import main


client = TestClient(main.app)


def rating_payload(**overrides):
    payload = {
        "game_id": "game-123",
        "engine": "stockfish",
        "fen": chess.STARTING_FEN,
        "move": "e4",
        "move_number": "1.",
        "commentary": "Controls the center",
        "quality": 4,
        "responsiveness": 5,
        "correctness": 4,
        "relevance": 4,
        "salience": 3,
        "review": "Good move",
    }
    return {**payload, **overrides}


def test_rate_move_appends_rating_jsonl(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "ratings.jsonl"
    monkeypatch.setenv("RATINGS_PATH", str(path))

    response = client.post("/api/rate_move", json=rating_payload())

    assert response.status_code == 200
    assert response.json() == {"status": "saved"}
    entry = json.loads(path.read_text().strip())
    assert {key: entry[key] for key in rating_payload()} == rating_payload()
    assert entry["timestamp"].endswith("+00:00")


def test_rate_move_preserves_existing_ratings(tmp_path, monkeypatch):
    path = tmp_path / "ratings.jsonl"
    monkeypatch.setenv("RATINGS_PATH", str(path))

    assert client.post("/api/rate_move", json=rating_payload(move="e4")).status_code == 200
    assert client.post("/api/rate_move", json=rating_payload(move="d4")).status_code == 200

    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert [entry["move"] for entry in entries] == ["e4", "d4"]


def test_rate_move_returns_sanitized_error_when_storage_fails(monkeypatch):
    def fail(rating):
        raise OSError("private storage details")

    monkeypatch.setattr(main, "save_rating", fail)
    response = TestClient(main.app, raise_server_exceptions=False).post(
        "/api/rate_move", json=rating_payload()
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


def test_rate_move_requires_all_ratings_in_range():
    for quality in (0, 6):
        response = client.post(
            "/api/rate_move",
            json=rating_payload(quality=quality),
        )
        assert response.status_code == 422


def test_rate_move_rejects_invalid_fen(tmp_path, monkeypatch):
    path = tmp_path / "ratings.jsonl"
    monkeypatch.setenv("RATINGS_PATH", str(path))

    response = client.post(
        "/api/rate_move",
        json=rating_payload(fen="not a FEN"),
    )

    assert response.status_code == 400
    assert not path.exists()


@pytest.mark.parametrize("field", ["move", "game_id", "engine"])
def test_rate_move_rejects_empty_identifying_fields(tmp_path, monkeypatch, field):
    path = tmp_path / "ratings.jsonl"
    monkeypatch.setenv("RATINGS_PATH", str(path))

    response = client.post("/api/rate_move", json=rating_payload(**{field: ""}))

    assert response.status_code == 422
    assert not path.exists()


def test_global_exception_handler_returns_json_response():
    response = asyncio.run(
        main.global_exception_handler(None, RuntimeError("hidden details"))
    )

    assert response.status_code == 500
    assert json.loads(response.body) == {"detail": "Internal server error"}
