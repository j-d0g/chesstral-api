from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import sys
import os
import asyncio
import chess
import logging

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import MoveRequest, MoveResponse, EvaluationRequest, EvaluationResponse

logger = logging.getLogger(__name__)


class ChessEngine(ABC):
    """Abstract base class for all chess engines"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self._last_error_message = ""
    
    @abstractmethod
    async def get_move(self, request: MoveRequest) -> MoveResponse:
        """Get a move from the engine given the current position"""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the engine (load models, connect to APIs, etc.)"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources"""
        pass
    
    async def _make_api_call(self, prompt: str, system_msg: str, temperature: float) -> str:
        """Make API call to the specific engine - can be overridden by each engine"""
        raise NotImplementedError(f"{self.name} does not implement _make_api_call yet")
    
    async def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        """Evaluate a position (optional, not all engines support this)"""
        raise NotImplementedError(f"{self.name} does not support position evaluation")
    
    def calculate_retry_temperature(self, base_temperature: float, attempt: int, max_attempts: int) -> float:
        """
        Calculate temperature for retry attempts, respecting user settings.
        
        Args:
            base_temperature: User-provided temperature from the frontend
            attempt: Current attempt number (0-based)
            max_attempts: Maximum number of attempts (unused but kept for API compatibility)
            
        Returns:
            Adjusted temperature for this attempt, capped at 1.0
        """
        retry_multiplier = 1.0 + (attempt * 0.1)  # Slight increase for each retry
        return min(base_temperature * retry_multiplier, 1.0)  # Cap at 1.0
    
    def build_prompt(self, board: chess.Board, pgn_moves: list) -> str:
        """
        Build a standardized prompt for chess engines.
        
        Args:
            board: Current chess board position
            pgn_moves: List of moves in PGN format
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Start with game context if we have PGN moves
        if pgn_moves and len(pgn_moves) > 0:
            # Format PGN moves nicely
            move_pairs = []
            for i in range(0, len(pgn_moves), 2):
                move_num = (i // 2) + 1
                white_move = pgn_moves[i]
                black_move = pgn_moves[i + 1] if i + 1 < len(pgn_moves) else ""
                if black_move:
                    move_pairs.append(f"{move_num}. {white_move} {black_move}")
                else:
                    move_pairs.append(f"{move_num}. {white_move}")
            
            prompt_parts.append("Game so far:")
            prompt_parts.append(" ".join(move_pairs))
            prompt_parts.append("")
        
        # Add current position info
        color = "White" if board.turn else "Black"
        prompt_parts.append(f"Current position: {color} to move")
        
        # Add FEN as backup context
        prompt_parts.append(f"FEN: {board.fen()}")
        
        # Add tactical context
        if board.is_check():
            prompt_parts.append("Note: The king is in check!")
        
        # Request the move
        prompt_parts.append(f"What is the best move for {color}?")
        
        return "\n".join(prompt_parts)
    
    def extract_and_validate_move(self, response: str, board: chess.Board) -> str:
        """
        Extract and validate a move from the engine response.
        
        Args:
            response: Raw response from the engine
            board: Current chess board position
            
        Returns:
            Valid move in SAN notation
            
        Raises:
            ValueError: If no valid move is found
        """
        # Clean the response
        response = response.strip()
        
        # Try to find a valid move in the response
        potential_moves = []
        
        # Split by various delimiters and try each part
        for delimiter in [' ', '\n', '\t', ',', '.', '!', '?', ':', ';']:
            potential_moves.extend(response.split(delimiter))
        
        # Also try the whole response
        potential_moves.append(response)
        
        # Get legal moves for error messages
        legal_moves = list(board.legal_moves)
        legal_moves_san = [board.san(move) for move in legal_moves]
        
        for potential_move in potential_moves:
            clean_move = potential_move.strip(".,!?()[]{}\"' \n\t")
            
            if not clean_move:
                continue
                
            try:
                # Try to parse as SAN first
                parsed_move = board.parse_san(clean_move)
                return clean_move  # Return the original SAN notation
                
            except chess.IllegalMoveError:
                # Try UCI format
                try:
                    uci_move = chess.Move.from_uci(clean_move)
                    if uci_move in legal_moves:
                        return board.san(uci_move)  # Convert to SAN
                except:
                    pass
                    
            except chess.AmbiguousMoveError:
                raise ValueError(f"Ambiguous move: '{clean_move}'. Please specify the file of the origin piece (e.g., Nhg8 instead of Ng8) or use UCI format. Legal moves: {', '.join(legal_moves_san[:10])}...")
                
            except chess.InvalidMoveError:
                continue  # Try next potential move
        
        # No valid move found in any part of the response
        raise ValueError(f"No valid move found in response: '{response}'. Please provide a legal move in standard algebraic notation. Legal moves: {', '.join(legal_moves_san[:10])}...")
    
    async def reprompt_loop(self, board: chess.Board, request: MoveRequest, max_retries: int = 5) -> Tuple[str, str, str]:
        """
        Standard reprompting loop with move validation and error feedback.
        
        Args:
            board: Current chess board position
            request: Move request with engine parameters
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (move, raw_response, thoughts)
        """
        for attempt in range(max_retries):
            try:
                # Build prompt for this attempt
                if attempt == 0:
                    prompt = self.build_prompt(board, request.pgn)
                    system_msg = "You are a chess grandmaster. Analyze the position and respond with only the best chess move in standard algebraic notation (e.g., e4, Nf3, O-O, Qxd5). Do not include explanations or commentary."
                else:
                    # For retries, include the error message in the prompt
                    prompt = f"{self._last_error_message}\n\nOriginal position:\n{self.build_prompt(board, request.pgn)}"
                    system_msg = "You made an error in your previous move. Please provide a valid chess move in standard algebraic notation."
                
                logger.info(f"{self.name} attempt {attempt + 1}/{max_retries}")
                
                # Use shared temperature calculation
                temperature = self.calculate_retry_temperature(request.temperature, attempt, max_retries)
                logger.info(f"🌡️  Using temperature: {temperature:.3f} (base: {request.temperature:.3f}, attempt {attempt + 1})")
                
                # Make the API call (implemented by each engine)  
                try:
                    raw_response = await self._make_api_call(prompt, system_msg, temperature)
                    logger.info(f"{self.name} raw response: {raw_response}")
                except NotImplementedError:
                    # Engine hasn't implemented _make_api_call yet, skip shared reprompt loop
                    raise Exception(f"{self.name} doesn't support shared reprompt loop yet")
                
                # Extract and validate move
                try:
                    move = self.extract_and_validate_move(raw_response, board)
                    # Success!
                    thoughts = f"{self.name} (attempt {attempt + 1}): {raw_response}"
                    return move, raw_response, thoughts
                    
                except ValueError as move_error:
                    # Invalid move - prepare error message for next attempt
                    self._last_error_message = str(move_error)
                    logger.warning(f"Attempt {attempt + 1} failed: {move_error}")
                    
                    if attempt == max_retries - 1:
                        # Last attempt failed
                        raise Exception(f"Failed to get valid move after {max_retries} attempts. Last error: {move_error}")
                    
                    # Continue to next attempt
                    continue
                    
            except asyncio.TimeoutError:
                logger.error(f"Attempt {attempt + 1} timed out")
                if attempt == max_retries - 1:
                    raise Exception(f"{self.name} API request timed out after multiple attempts")
                continue
                
        raise Exception(f"Failed to get valid move after {max_retries} attempts") 