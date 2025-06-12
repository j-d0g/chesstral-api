"""
Comprehensive error handling service for ChessGPT API
Provides detailed error tracking, user-friendly messages, and recovery suggestions
"""

import time
import traceback
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import chess
from core.types import MoveAttempt, MoveError, ErrorResponse
import logging

logger = logging.getLogger(__name__)


class MoveErrorTracker:
    """Tracks move attempts and errors for detailed reporting"""
    
    def __init__(self, max_retries: int = 5):
        self.max_retries = max_retries
        self.attempts: List[MoveAttempt] = []
        self.start_time = time.time()
        
    def add_attempt(self, 
                   attempt_number: int,
                   attempted_move: Optional[str] = None,
                   error_type: Optional[str] = None,
                   error_message: Optional[str] = None,
                   raw_response: Optional[str] = None,
                   board: Optional[chess.Board] = None) -> MoveAttempt:
        """Add a move attempt with error details"""
        
        # Get sample of legal moves for context
        legal_moves_sample = None
        if board and error_type in ["illegal_move", "invalid_format"]:
            legal_moves = [board.san(move) for move in list(board.legal_moves)]
            legal_moves_sample = legal_moves[:8]  # Show first 8 legal moves
            
        attempt = MoveAttempt(
            attempt_number=attempt_number,
            attempted_move=attempted_move,
            error_type=error_type,
            error_message=error_message,
            legal_moves_sample=legal_moves_sample,
            raw_response=raw_response[:200] + "..." if raw_response and len(raw_response) > 200 else raw_response,
            timestamp=datetime.now().isoformat()
        )
        
        self.attempts.append(attempt)
        return attempt
        
    def get_failed_attempts(self) -> List[MoveAttempt]:
        """Get all failed attempts"""
        return [attempt for attempt in self.attempts if attempt.error_type is not None]
        
    def get_response_time_ms(self) -> int:
        """Get total response time in milliseconds"""
        return int((time.time() - self.start_time) * 1000)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all attempts"""
        failed_attempts = self.get_failed_attempts()
        return {
            "total_attempts": len(self.attempts),
            "failed_attempts": len(failed_attempts),
            "success_rate": (len(self.attempts) - len(failed_attempts)) / max(len(self.attempts), 1),
            "response_time_ms": self.get_response_time_ms(),
            "error_types": list(set(attempt.error_type for attempt in failed_attempts if attempt.error_type))
        }


class ChessErrorHandler:
    """Comprehensive error handling for chess move generation"""
    
    @staticmethod
    def classify_move_error(error: Exception, attempted_move: str, board: chess.Board) -> Tuple[str, str]:
        """Classify the type of move error and provide a clear message"""
        
        if isinstance(error, chess.IllegalMoveError):
            return "illegal_move", f"The move '{attempted_move}' is not legal in the current position"
            
        elif isinstance(error, chess.AmbiguousMoveError):
            return "ambiguous_move", f"The move '{attempted_move}' is ambiguous - please specify the origin square"
            
        elif isinstance(error, chess.InvalidMoveError):
            return "invalid_format", f"The move '{attempted_move}' has an invalid format"
            
        elif isinstance(error, ValueError):
            if "no valid move" in str(error).lower():
                return "no_valid_move", f"No valid chess move found in the response"
            else:
                return "parse_error", f"Could not parse the move: {str(error)}"
                
        elif isinstance(error, TimeoutError) or "timeout" in str(error).lower():
            return "timeout", f"The chess engine took too long to respond"
            
        else:
            return "unknown_error", f"Unexpected error: {str(error)}"
    
    @staticmethod
    def get_helpful_suggestions(error_type: str, board: chess.Board, engine_type: str) -> List[str]:
        """Get helpful suggestions based on the error type"""
        
        suggestions = []
        
        # Handle None board case
        if board is None:
            if error_type == "missing_data":
                suggestions.extend([
                    "Ensure all required fields are provided",
                    "Check that the engine, model, and FEN are specified",
                    "Verify the request format is correct"
                ])
            else:
                suggestions.extend([
                    "Unable to provide specific move suggestions without a valid board position",
                    "Please ensure a valid FEN position is provided"
                ])
            return suggestions
        
        legal_moves = [board.san(move) for move in list(board.legal_moves)]
        
        if error_type == "illegal_move":
            suggestions.extend([
                f"Try one of these legal moves: {', '.join(legal_moves[:5])}{'...' if len(legal_moves) > 5 else ''}",
                "Check if you're trying to move the right piece",
                "Verify the current board position is correct"
            ])
            
        elif error_type == "invalid_format":
            suggestions.extend([
                "Use standard algebraic notation (e.g., e4, Nf3, O-O)",
                "Avoid extra characters or formatting",
                f"Legal moves in this position: {', '.join(legal_moves[:3])}..."
            ])
            
        elif error_type == "timeout":
            suggestions.extend([
                "The AI engine may be overloaded - try again in a moment",
                "Consider switching to a faster engine like NanoGPT",
                "Check your internet connection"
            ])
            
        elif error_type == "ambiguous_move":
            suggestions.extend([
                "Specify the origin square (e.g., Nbd2 instead of Nd2)",
                "Use long algebraic notation if needed",
                "Try UCI format (e.g., g1f3 instead of Nf3)"
            ])
            
        elif error_type == "no_valid_move":
            suggestions.extend([
                f"The AI response didn't contain a recognizable move",
                f"Legal moves are: {', '.join(legal_moves[:5])}{'...' if len(legal_moves) > 5 else ''}",
                "Try a different AI engine or adjust the temperature"
            ])
            
        # Engine-specific suggestions
        if engine_type == "nanogpt":
            suggestions.append("NanoGPT works best in standard opening positions")
        elif engine_type in ["openai", "anthropic", "gemini", "deepseek"]:
            suggestions.append("Try adjusting the temperature setting if moves seem random")
            
        return suggestions
    
    @staticmethod
    def get_recovery_options(error_type: str, total_attempts: int, max_retries: int) -> List[str]:
        """Get recovery options for the user"""
        
        options = []
        
        if total_attempts < max_retries:
            options.append(f"The system will automatically retry ({total_attempts}/{max_retries} attempts used)")
            
        if error_type in ["timeout", "api_error"]:
            options.extend([
                "Wait a moment and try again",
                "Switch to a different AI engine",
                "Check if the API service is available"
            ])
            
        elif error_type in ["illegal_move", "invalid_format"]:
            options.extend([
                "The AI will be prompted with the legal moves",
                "Consider adjusting the position if it seems incorrect",
                "Try a different AI engine if problems persist"
            ])
            
        if total_attempts >= max_retries:
            options.extend([
                "Maximum retries reached - try a different engine",
                "Reset the game position if it seems corrupted",
                "Contact support if the issue persists"
            ])
            
        return options
    
    @staticmethod
    def create_move_error(error_type: str, 
                         error_message: str,
                         attempted_move: Optional[str] = None,
                         board: Optional[chess.Board] = None,
                         total_attempts: int = 0,
                         failed_attempts: List[MoveAttempt] = None,
                         engine_type: str = "unknown") -> MoveError:
        """Create a comprehensive MoveError object"""
        
        if failed_attempts is None:
            failed_attempts = []
            
        legal_moves = None
        if board is not None:
            legal_moves = [board.san(move) for move in list(board.legal_moves)]
            
        suggestions = ChessErrorHandler.get_helpful_suggestions(error_type, board, engine_type)
        recovery_options = ChessErrorHandler.get_recovery_options(error_type, total_attempts, 5)
        
        return MoveError(
            error_type=error_type,
            error_message=error_message,
            attempted_move=attempted_move,
            legal_moves=legal_moves,
            total_attempts=total_attempts,
            failed_attempts=failed_attempts,
            suggestions=suggestions,
            recovery_options=recovery_options
        )
    
    @staticmethod
    def create_error_response(error: Exception,
                            error_context: str = "",
                            move_error: Optional[MoveError] = None) -> ErrorResponse:
        """Create a comprehensive error response"""
        
        error_message = str(error)
        error_type = "general_error"
        
        # Classify common error types
        if isinstance(error, chess.IllegalMoveError):
            error_type = "illegal_move"
        elif isinstance(error, TimeoutError):
            error_type = "timeout"
        elif "api" in error_message.lower():
            error_type = "api_error"
        elif "network" in error_message.lower():
            error_type = "network_error"
            
        suggestions = []
        recovery_options = []
        
        if error_type == "timeout":
            suggestions.extend([
                "The request took too long to complete",
                "Try a faster AI engine or reduce complexity"
            ])
            recovery_options.extend([
                "Wait a moment and try again",
                "Switch to NanoGPT for faster responses"
            ])
            
        elif error_type == "api_error":
            suggestions.extend([
                "There was an issue with the AI service",
                "Check if your API keys are configured correctly"
            ])
            recovery_options.extend([
                "Try a different AI engine",
                "Check the server logs for more details"
            ])
            
        elif error_type == "network_error":
            suggestions.extend([
                "Could not connect to the AI service",
                "Check your internet connection"
            ])
            recovery_options.extend([
                "Verify the backend server is running",
                "Try again in a few moments"
            ])
            
        return ErrorResponse(
            error=f"{error_context}: {error_message}" if error_context else error_message,
            error_type=error_type,
            details=traceback.format_exc() if logger.level <= 10 else None,  # Only include traceback in debug mode
            move_error=move_error,
            suggestions=suggestions,
            recovery_options=recovery_options,
            timestamp=datetime.now().isoformat()
        )


def log_move_attempt(tracker: MoveErrorTracker, engine_name: str, board: chess.Board):
    """Log detailed information about move attempts"""
    
    summary = tracker.get_summary()
    failed_attempts = tracker.get_failed_attempts()
    
    logger.info(f"🎯 Move generation summary for {engine_name}:")
    logger.info(f"   Total attempts: {summary['total_attempts']}")
    logger.info(f"   Failed attempts: {summary['failed_attempts']}")
    logger.info(f"   Success rate: {summary['success_rate']:.1%}")
    logger.info(f"   Response time: {summary['response_time_ms']}ms")
    
    if failed_attempts:
        logger.warning(f"❌ Failed attempts details:")
        for attempt in failed_attempts:
            logger.warning(f"   Attempt {attempt.attempt_number}: {attempt.error_type} - {attempt.attempted_move}")
            if attempt.legal_moves_sample:
                logger.warning(f"      Legal moves: {', '.join(attempt.legal_moves_sample)}")
                
    # Log position context for debugging
    logger.debug(f"🏁 Position: {board.fen()}")
    logger.debug(f"🎲 Legal moves: {[board.san(move) for move in list(board.legal_moves)]}") 