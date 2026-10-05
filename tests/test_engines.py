import asyncio

import chess
import pytest

from core.types import EngineType, MoveRequest
from engines.base import ChessEngine
from engines.deepseek_engine import DeepSeekEngine
from engines.gemini_engine import GeminiEngine


def test_extract_and_validate_move():
    engine = DeepSeekEngine()
    board = chess.Board()

    for response, expected in [
        ("e4", "e4"),
        ("I play Nf3.", "Nf3"),
        ("e2e4", "e4"),
        ("The best move is d4", "d4"),
    ]:
        assert engine.extract_and_validate_move(response, board) == expected

    with pytest.raises(ValueError):
        engine.extract_and_validate_move("garbage", board)


def test_reprompt_loop_retries_invalid_move():
    class ScriptedEngine(ChessEngine):
        def __init__(self):
            super().__init__("scripted")
            self.responses = ["xyz", "e4"]
            self.calls = []

        async def get_move(self, request):
            raise NotImplementedError

        async def initialize(self):
            pass

        async def shutdown(self):
            pass

        async def _make_api_call(self, prompt, system_msg, temperature):
            self.calls.append((prompt, system_msg, temperature))
            return self.responses.pop(0)

    engine = ScriptedEngine()
    request = MoveRequest(fen=chess.Board().fen(), engine=EngineType.DEEPSEEK)

    move, _, _ = asyncio.run(engine.reprompt_loop(chess.Board(), request))

    assert move == "e4"
    assert len(engine.calls) == 2
    assert engine._last_error_message in engine.calls[1][0]
    assert engine.calls[1][2] > engine.calls[0][2]


def test_deepseek_get_move_uses_shared_reprompt_loop(monkeypatch):
    engine = DeepSeekEngine()
    engine.available = True
    engine.api_key = "test"

    async def make_api_call(prompt, system_msg, temperature):
        return "e4"

    monkeypatch.setattr(engine, "_make_api_call", make_api_call)
    request = MoveRequest(fen=chess.Board().fen(), engine=EngineType.DEEPSEEK)

    response = asyncio.run(engine.get_move(request))

    assert response.move == "e4"


def test_gemini_get_move_uses_shared_reprompt_loop(monkeypatch):
    engine = GeminiEngine()
    engine.available = True
    engine.client = object()

    async def make_api_call(prompt, system_msg, temperature):
        return "e4"

    monkeypatch.setattr(engine, "_make_api_call", make_api_call)
    request = MoveRequest(fen=chess.Board().fen(), engine=EngineType.GEMINI)

    response = asyncio.run(engine.get_move(request))

    assert response.move == "e4"
    assert engine.build_prompt(chess.Board(), []).endswith(
        " Respond with only the move in standard algebraic notation (e.g., e4, Nf3, O-O, Qxd5)."
    )


def test_gemini_build_prompt_matches_legacy_prompt():
    engine = GeminiEngine()
    board = chess.Board()
    suffix = " Respond with only the move in standard algebraic notation (e.g., e4, Nf3, O-O, Qxd5)."

    assert engine.build_prompt(board, []) == (
        f"Current position: White to move\nFEN: {board.fen()}\n"
        f"What is the best move for White?{suffix}"
    )
