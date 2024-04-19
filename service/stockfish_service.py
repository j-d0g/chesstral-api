import chess.engine
from flask import jsonify


def get_stockfish_move(fen, stockfish_path):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    board = chess.Board(fen)

    move = engine.play(board, chess.engine.Limit(time=0.1)).move
    commentary = f"Stockfish plays {move}"
    return jsonify({'move': str(move), 'commentary': commentary})
