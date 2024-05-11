import chess.engine
import os
from flask import jsonify
from dotenv import load_dotenv

from util.chess_features import to_san

load_dotenv()


class Stockfish:
    def __init__(self):
        self.stockfish_path = os.getenv('STOCKFISH_PATH')
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)

    def get_stockfish_move(self, fen):
        board = chess.Board(fen)
        result = self.engine.play(board, chess.engine.Limit(time=0.1))
        move = result.move
        commentary = f"Stockfish plays {move}"
        return {'move': str(move), 'commentary': commentary}

    def get_stockfish_evaluation(self, fen):
        board = chess.Board(fen)
        info = self.engine.analyse(board, chess.engine.Limit(time=0.1))
        score = info['score'].relative.score(mate_score=10000)  # mate_score optional for clarity on mate situations
        # Convert score to a more readable format: positive for white's advantage, negative for black's
        if score is not None:
            readable_score = score / 100.0  # converting centi pawns to pawns
        else:
            readable_score = "Mate imminent"
        return {'evaluation': readable_score}


# Example Usage:

# Create an instance of the Stockfish class
stockfish = Stockfish()
# Get the best move and evaluation for the initial position
board = chess.Board()
fen = board.fen()
evaluation_response = stockfish.get_stockfish_evaluation(fen)
print(evaluation_response)

move_response = stockfish.get_stockfish_move(fen)
board.push(chess.Move.from_uci(move_response['move']))
fen = board.fen()
evaluation_response = stockfish.get_stockfish_evaluation(fen)
print(move_response['move'])
print(evaluation_response)