from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel
from enum import Enum


class PlayerColor(str, Enum):
    WHITE = "white"
    BLACK = "black"


class EngineType(str, Enum):
    NANOGPT = "nanogpt"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    REPLICATE = "replicate"
    STOCKFISH = "stockfish"


class MoveAttempt(BaseModel):
    """Details of a single move attempt"""
    attempt_number: int
    attempted_move: Optional[str] = None
    error_type: Optional[str] = None  # "illegal_move", "invalid_format", "parse_error", "timeout", "api_error"
    error_message: Optional[str] = None
    legal_moves_sample: Optional[List[str]] = None  # Sample of legal moves for context
    raw_response: Optional[str] = None
    timestamp: Optional[str] = None


class MoveRequest(BaseModel):
    """Request for getting a move from an engine"""
    fen: str
    pgn: List[str] = []
    engine: EngineType
    model: Optional[str] = None  # e.g., "gpt-4", "claude-3", "small-16"
    temperature: float = 0.7
    context: Dict[str, Any] = {}


class MoveResponse(BaseModel):
    """Response containing the engine's move with detailed error tracking"""
    move: str
    thoughts: Optional[str] = None
    raw_response: Optional[str] = None
    evaluation: Optional[float] = None
    
    # Enhanced error tracking
    total_attempts: int = 1
    successful_attempt: int = 1
    failed_attempts: List[MoveAttempt] = []
    warnings: List[str] = []
    
    # Performance metrics
    response_time_ms: Optional[int] = None
    engine_status: str = "success"  # "success", "partial_success", "fallback_used", "failed"


class MoveError(BaseModel):
    """Detailed error information for move failures"""
    error_type: str  # "illegal_move", "invalid_format", "engine_unavailable", "timeout", "max_retries_exceeded"
    error_message: str
    attempted_move: Optional[str] = None
    legal_moves: Optional[List[str]] = None
    total_attempts: int = 0
    failed_attempts: List[MoveAttempt] = []
    suggestions: List[str] = []  # Helpful suggestions for the user
    recovery_options: List[str] = []  # What the user can try next
    

class MoveRatingRequest(BaseModel):
    """Request for rating a move"""
    fen: str
    move: str
    rating: int
    context: Dict[str, Any] = {}


class EvaluationRequest(BaseModel):
    """Request for board evaluation"""
    fen: str
    depth: int = 10


class EvaluationResponse(BaseModel):
    """Response containing board evaluation"""
    evaluation: float
    best_move: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    

class GameState(BaseModel):
    """Current game state"""
    fen: str
    pgn: List[str]
    turn: PlayerColor
    is_game_over: bool = False
    result: Optional[str] = None


class ErrorResponse(BaseModel):
    """Enhanced error response with detailed information"""
    error: str
    error_type: str = "general_error"
    details: Optional[str] = None 
    move_error: Optional[MoveError] = None
    suggestions: List[str] = []
    recovery_options: List[str] = []
    timestamp: Optional[str] = None 