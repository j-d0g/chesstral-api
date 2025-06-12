import chess
import chess.engine
from typing import Dict, Any

from .base import ChessEngine
from core.types import MoveRequest, MoveResponse, EvaluationRequest, EvaluationResponse


class StockfishEngine(ChessEngine):
    """Stockfish chess engine implementation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("stockfish", config)
        self.engine = None
    
    async def initialize(self) -> None:
        """Initialize Stockfish engine"""
        try:
            # Try common Stockfish paths
            stockfish_paths = [
                "stockfish",
                "/usr/local/bin/stockfish", 
                "/opt/homebrew/bin/stockfish",
                "/usr/bin/stockfish"
            ]
            
            for path in stockfish_paths:
                try:
                    self.engine = chess.engine.SimpleEngine.popen_uci(path)
                    break
                except FileNotFoundError:
                    continue
            
            if not self.engine:
                raise FileNotFoundError("Stockfish not found. Please install Stockfish.")
                
        except Exception as e:
            raise Exception(f"Failed to initialize Stockfish: {str(e)}")
    
    async def shutdown(self) -> None:
        """Clean up Stockfish engine"""
        if self.engine:
            self.engine.quit()
    
    async def get_move(self, request: MoveRequest) -> MoveResponse:
        """Get best move from Stockfish"""
        try:
            board = chess.Board(request.fen)
            
            # Get best move from Stockfish
            result = self.engine.play(board, chess.engine.Limit(time=1.0))
            best_move = result.move
            
            return MoveResponse(
                move=board.san(best_move),
                thoughts=f"Stockfish best move: {board.san(best_move)}"
            )
            
        except Exception as e:
            raise Exception(f"Stockfish engine error: {str(e)}")
    
    async def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        """Evaluate position with Stockfish"""
        try:
            board = chess.Board(request.fen)
            
            # Analyze position
            info = self.engine.analyse(board, chess.engine.Limit(depth=request.depth))
            
            # Get evaluation score
            score = info["score"].relative
            if score.is_mate():
                eval_score = 1000.0 if score.mate() > 0 else -1000.0
            else:
                eval_score = float(score.score()) / 100.0  # Convert centipawns to pawns
            
            # Get best move
            best_move = None
            if "pv" in info and info["pv"]:
                best_move = board.san(info["pv"][0])
            
            return EvaluationResponse(
                evaluation=eval_score,
                best_move=best_move,
                analysis={
                    "depth": info.get("depth", 0),
                    "nodes": info.get("nodes", 0),
                    "time": info.get("time", 0),
                }
            )
            
        except Exception as e:
            raise Exception(f"Stockfish evaluation error: {str(e)}") 