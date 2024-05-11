import chess.engine
from flask import jsonify
from dotenv import load_dotenv
import os

load_dotenv()


class Stockfish:
    def __init__(self):
        self.stockfish_path = os.getenv('STOCKFISH_PATH')
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)

    def get_stockfish_move(self, fen):
        board = chess.Board(fen)

        move = self.engine.play(board, chess.engine.Limit(time=0.1)).move
        commentary = f"Stockfish plays {move}"
        return jsonify({'move': str(move), 'commentary': commentary})

    def get_stockfish_eval(self, fen):
        board = chess.Board(fen)

        info = self.engine.analyse(board, chess.engine.Limit(time=0.1))
        return jsonify({'evaluation': str(info['score'])})


# test get_stockfish_eval
chess.board = chess.Board()
stockfish_path = 'stockfish'
