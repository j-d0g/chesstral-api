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
            
            # Analyze position with more time for better analysis
            time_limit = max(2.0, request.depth * 0.1)  # Scale time with depth
            
            # Use multipv parameter directly in analyse call (don't configure manually)
            info = self.engine.analyse(
                board, 
                chess.engine.Limit(depth=request.depth, time=time_limit), 
                multipv=3
            )
            
            # Get primary evaluation score
            primary_info = info[0] if isinstance(info, list) else info
            score = primary_info["score"].relative
            
            if score.is_mate():
                eval_score = 1000.0 if score.mate() > 0 else -1000.0
            else:
                eval_score = float(score.score()) / 100.0  # Convert centipawns to pawns
            
            # Get best move
            best_move = None
            if "pv" in primary_info and primary_info["pv"]:
                best_move = board.san(primary_info["pv"][0])
            
            # Build detailed analysis
            analysis_data = {
                "depth": primary_info.get("depth", 0),
                "nodes": primary_info.get("nodes", 0),
                "time": int(primary_info.get("time", 0) * 1000),  # Convert to milliseconds
            }
            
            # Add principal variation
            if "pv" in primary_info and primary_info["pv"]:
                pv_moves = []
                temp_board = board.copy()
                for move in primary_info["pv"][:20]:  # Limit to first 20 moves
                    try:
                        pv_moves.append(temp_board.san(move))
                        temp_board.push(move)
                    except:
                        break
                analysis_data["pv"] = pv_moves
            
            # Add multiple PV lines if available
            if isinstance(info, list) and len(info) > 1:
                multipv_lines = []
                for pv_info in info:
                    if "pv" in pv_info and pv_info["pv"]:
                        pv_score = pv_info["score"].relative
                        if pv_score.is_mate():
                            pv_eval = 1000.0 if pv_score.mate() > 0 else -1000.0
                        else:
                            pv_eval = float(pv_score.score()) / 100.0
                        
                        # Get PV moves for this line
                        pv_moves = []
                        temp_board = board.copy()
                        for move in pv_info["pv"][:10]:  # Limit to first 10 moves per line
                            try:
                                pv_moves.append(temp_board.san(move))
                                temp_board.push(move)
                            except:
                                break
                        
                        multipv_lines.append({
                            "move": board.san(pv_info["pv"][0]),
                            "evaluation": pv_eval,
                            "pv": pv_moves
                        })
                
                analysis_data["multipv"] = multipv_lines
            
            return EvaluationResponse(
                evaluation=eval_score,
                best_move=best_move,
                analysis=analysis_data
            )
            
        except Exception as e:
            raise Exception(f"Stockfish evaluation error: {str(e)}") 