import asyncio
import shutil
from pathlib import Path

import chess
import pytest

from core.types import EvaluationRequest
from engines.stockfish_engine import StockfishEngine


STOCKFISH = shutil.which("stockfish") or next(
    (
        str(path)
        for path in (
            Path("/opt/homebrew/bin/stockfish"),
            Path("/usr/local/bin/stockfish"),
            Path("/usr/bin/stockfish"),
        )
        if path.is_file()
    ),
    None,
)
pytestmark = pytest.mark.skipif(STOCKFISH is None, reason="Stockfish is not installed")


@pytest.fixture(scope="module")
def stockfish():
    engine = StockfishEngine()
    asyncio.run(engine.initialize())
    yield engine
    asyncio.run(engine.shutdown())


def evaluate(engine, fen):
    return asyncio.run(engine.evaluate(EvaluationRequest(fen=fen, depth=10)))


def test_eval_after_e4_is_from_white_perspective(stockfish):
    board = chess.Board()
    board.push_san("e4")

    response = evaluate(stockfish, board.fen())

    assert response.evaluation > 0


def test_eval_of_white_winning_position_is_positive_with_black_to_move(stockfish):
    response = evaluate(
        stockfish,
        "6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1",
    )

    assert response.evaluation > 3


def test_white_mate_in_one_has_positive_mate_distance(stockfish):
    response = evaluate(stockfish, "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")

    assert response.mate == 1
    assert response.evaluation == 1000.0


@pytest.mark.parametrize(
    ("moves", "expected_evaluation"),
    [
        (["f3", "e5", "g4", "Qh4#"], -1000.0),
        (["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"], 1000.0),
    ],
)
def test_checkmated_side_has_zero_mate_distance(stockfish, moves, expected_evaluation):
    board = chess.Board()
    for move in moves:
        board.push_san(move)
    assert board.is_checkmate()

    response = evaluate(stockfish, board.fen())

    assert response.mate == 0
    assert response.evaluation == expected_evaluation
