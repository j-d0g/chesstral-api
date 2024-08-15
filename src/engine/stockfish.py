import chess.engine
import os
from dotenv import load_dotenv

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

        return {
            'uuid': '0',
            'prompt': {
                'completion': {
                    'move': str(move),
                    'thoughts': commentary
                },
                'context': []
            },
        }

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
