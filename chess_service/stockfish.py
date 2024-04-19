import chess.engine
from flask import jsonify
from config import STOCKFISH_PATH


def get_stockfish_move(fen):
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    board = chess.Board(fen)

    move = engine.play(board, chess.engine.Limit(time=0.1)).move
    commentary = f"Stockfish plays {move}"
    return jsonify({'move': str(move), 'commentary': commentary})
