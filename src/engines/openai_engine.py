import os
import chess
from typing import Dict, Any
import asyncio
import logging

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from .base import ChessEngine
from core.types import MoveRequest, MoveResponse

logger = logging.getLogger(__name__)


class OpenAIEngine(ChessEngine):
    """OpenAI chess engine implementation"""
    
    def __init__(self, model: str = "o1-mini", config: Dict[str, Any] = None):
        super().__init__(f"openai-{model}", config)
        self.model = model
        self.client = None
        self.available = False
        self._last_error_message = ""
    
    async def initialize(self) -> None:
        """Initialize the OpenAI client"""
        if AsyncOpenAI is None:
            logger.error("OpenAI library not installed. Install with: pip install openai")
            return
            
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            return
        
        try:
            self.client = AsyncOpenAI(api_key=api_key)
            # Test the connection with a simple request
            await self._test_connection()
            self.available = True
            logger.info(f"OpenAI engine initialized successfully with model {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.available = False
    
    async def _test_connection(self) -> None:
        """Test the OpenAI API connection"""
        try:
            # o1 models don't support system messages and use different parameters
            if self.model.startswith('o1'):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Test"}]
                )
            else:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=1
                )
            logger.info("OpenAI API connection test successful")
        except Exception as e:
            raise Exception(f"OpenAI API connection test failed: {e}")
    
    async def shutdown(self) -> None:
        """Clean up resources"""
        if self.client:
            await self.client.close()
    
    async def get_move(self, request: MoveRequest) -> MoveResponse:
        """Get a move from OpenAI with proper reprompting for invalid moves"""
        if not self.available or not self.client:
            raise Exception("OpenAI engine not available. Check API key and connection.")
            
        try:
            # Create board from FEN
            board = chess.Board(request.fen)
            
            # Use reprompting loop for robust move generation
            move, raw_response, thoughts = await self._reprompt_loop(board, request)
            
            return MoveResponse(
                move=move,
                raw_response=raw_response,
                thoughts=thoughts
            )
            
        except Exception as e:
            logger.error(f"OpenAI engine error: {e}")
            raise Exception(f"OpenAI engine error: {str(e)}")
    
    async def _reprompt_loop(self, board: chess.Board, request: MoveRequest, max_retries: int = 5) -> tuple[str, str, str]:
        """Enhanced reprompting loop with comprehensive error tracking"""
        from service.error_handler import MoveErrorTracker, ChessErrorHandler
        
        # Initialize error tracking
        error_tracker = MoveErrorTracker(max_retries)
        
        for attempt in range(max_retries):
            try:
                # Build prompt for this attempt
                if attempt == 0:
                    prompt = self._build_prompt(board, request.pgn)
                    if self.model.startswith('o1'):
                        # o1 models work best with direct instructions
                        system_msg = """You are a chess grandmaster. Analyze the position and provide:
1. Your thoughts about the position and your reasoning
2. The best move in standard algebraic notation

Format your response as:
THOUGHTS: [Your analysis and reasoning here]
MOVE: [Your move in SAN notation]"""
                    else:
                        system_msg = """You are a chess grandmaster. Analyze the position and respond with your thoughts and the best move.
Format your response as JSON:
{
  "thoughts": "Your analysis and reasoning about the position",
  "move": "Your move in standard algebraic notation"
}"""
                else:
                    # For retries, include specific error feedback
                    failed_attempts = error_tracker.get_failed_attempts()
                    if failed_attempts:
                        last_attempt = failed_attempts[-1]
                        if last_attempt.error_type == "illegal_move":
                            error_context = f"Your previous move '{last_attempt.attempted_move}' was ILLEGAL. Legal moves: {', '.join(last_attempt.legal_moves_sample or [])}"
                        elif last_attempt.error_type == "invalid_format":
                            error_context = f"Your previous response had invalid format. Provide ONLY a chess move in standard notation."
                        elif last_attempt.error_type == "timeout":
                            error_context = "Your previous response timed out. Please respond quickly with just a move."
                        else:
                            error_context = f"Your previous response was invalid: {last_attempt.error_message}"
                    else:
                        error_context = "Your previous response was invalid. Please provide a legal chess move."
                    
                    prompt = f"{error_context}\n\nOriginal position:\n{self._build_prompt(board, request.pgn)}"
                    system_msg = "You made an error in your previous move. Please provide a valid chess move in standard algebraic notation."
                
                logger.info(f"🔄 OpenAI attempt {attempt + 1}/{max_retries}")
                
                # Calculate temperature using the evaluation system's strategy
                eval_temperature = min(((attempt / max_retries) * 1) + 0.001, 0.5)
                logger.info(f"🌡️  Using temperature: {eval_temperature:.3f} for attempt {attempt + 1}")
                
                # Get response from OpenAI
                # o1 models don't support system messages or temperature
                if self.model.startswith('o1'):
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "user", "content": f"{system_msg}\n\n{prompt}"}
                            ],
                            timeout=30.0
                        ),
                        timeout=35.0
                    )
                else:
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": system_msg},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=eval_temperature,
                            max_tokens=50,
                            timeout=30.0
                        ),
                        timeout=35.0
                    )
                
                raw_response = response.choices[0].message.content.strip()
                logger.info(f"📝 OpenAI raw response: {raw_response}")
                
                # Extract and validate move
                try:
                    # Extract move and thoughts based on model type
                    if self.model.startswith('o1'):
                        move, thoughts = self._extract_move_and_thoughts_o1(raw_response, board)
                    else:
                        move, thoughts = self._extract_move_and_thoughts_json(raw_response, board)
                    
                    # Track successful attempt
                    error_tracker.add_attempt(
                        attempt_number=attempt + 1,
                        attempted_move=move,
                        raw_response=raw_response
                    )
                    
                    # Success!
                    if attempt > 0:
                        thoughts += f" [Recovered after {attempt} failed attempts]"
                    
                    return move, raw_response, thoughts
                    
                except ValueError as move_error:
                    logger.warning(f"❌ Attempt {attempt + 1} failed: {move_error}")
                    
                    # Classify and track the error
                    error_type, _ = ChessErrorHandler.classify_move_error(move_error, raw_response, board)
                    
                    # Extract attempted move from error or response
                    attempted_move = self._extract_attempted_move(str(move_error), raw_response)
                    
                    error_tracker.add_attempt(
                        attempt_number=attempt + 1,
                        attempted_move=attempted_move,
                        error_type=error_type,
                        error_message=str(move_error),
                        raw_response=raw_response,
                        board=board
                    )
                    
                    if attempt == max_retries - 1:
                        # Create comprehensive error for final failure
                        failed_attempts = error_tracker.get_failed_attempts()
                        summary = error_tracker.get_summary()
                        
                        error_details = {
                            "engine": "openai",
                            "model": self.model,
                            "total_attempts": max_retries,
                            "failed_attempts": len(failed_attempts),
                            "error_types": summary["error_types"],
                            "last_error": str(move_error),
                            "response_time_ms": error_tracker.get_response_time_ms()
                        }
                        
                        raise Exception(f"OpenAI engine failed after {max_retries} attempts. Error details: {error_details}")
                    
                    # Continue to next attempt
                    continue
                    
            except asyncio.TimeoutError:
                logger.error(f"⏰ Attempt {attempt + 1} timed out")
                
                error_tracker.add_attempt(
                    attempt_number=attempt + 1,
                    error_type="timeout",
                    error_message="API request timed out after 35 seconds",
                    board=board
                )
                
                if attempt == max_retries - 1:
                    failed_attempts = error_tracker.get_failed_attempts()
                    timeout_count = sum(1 for a in failed_attempts if a.error_type == "timeout")
                    
                    error_details = {
                        "engine": "openai",
                        "model": self.model,
                        "total_attempts": max_retries,
                        "timeout_attempts": timeout_count,
                        "response_time_ms": error_tracker.get_response_time_ms()
                    }
                    
                    raise Exception(f"OpenAI API timed out after {max_retries} attempts. Error details: {error_details}")
                continue
                
        # This should never be reached due to the exception handling above
        raise Exception(f"Failed to get valid move after {max_retries} attempts")
    
    def _extract_attempted_move(self, error_message: str, raw_response: str) -> str:
        """Extract the attempted move from error message or raw response"""
        import re
        
        # Try to extract move from error message first
        move_match = re.search(r"'([^']+)'", error_message)
        if move_match:
            return move_match.group(1)
        
        # Try to extract from raw response
        potential_moves = raw_response.split()
        for word in potential_moves:
            clean_word = word.strip(".,!?()[]{}\"' \n\t")
            if len(clean_word) >= 2 and len(clean_word) <= 6:
                # Basic chess move pattern check
                if re.match(r'^[a-h1-8NBRQKO\-+#=]+$', clean_word):
                    return clean_word
        
        return raw_response[:20] + "..." if len(raw_response) > 20 else raw_response
    
    def _extract_and_validate_move(self, response: str, board: chess.Board) -> str:
        """Extract and validate a move from the response, with specific error messages"""
        
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
    
    def _extract_move(self, response: str, board: chess.Board) -> str:
        """Legacy method - now calls the new validation method"""
        return self._extract_and_validate_move(response, board)
    
    def _build_prompt(self, board: chess.Board, pgn_moves: list) -> str:
        """Build the prompt for OpenAI, emphasizing PGN context"""
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

    def _extract_move_and_thoughts_o1(self, raw_response: str, board: chess.Board) -> tuple[str, str]:
        """Extract move and thoughts from o1 format"""
        import re
        
        # Look for MOVE: pattern
        move_match = re.search(r"MOVE:\s*([^\n]+)", raw_response, re.IGNORECASE)
        if not move_match:
            # Fallback to old extraction method
            move = self._extract_and_validate_move(raw_response, board)
            thoughts = raw_response
            return move, thoughts
            
        potential_move = move_match.group(1).strip()
        
        # Look for THOUGHTS: pattern
        thoughts_match = re.search(r"THOUGHTS:\s*(.+?)(?=MOVE:|$)", raw_response, re.IGNORECASE | re.DOTALL)
        thoughts = thoughts_match.group(1).strip() if thoughts_match else "No thoughts provided"
        
        # Validate the move
        move = self._extract_and_validate_move(potential_move, board)
        
        return move, thoughts

    def _extract_move_and_thoughts_json(self, raw_response: str, board: chess.Board) -> tuple[str, str]:
        """Extract move and thoughts from JSON format"""
        import json
        
        try:
            # Try to parse as JSON
            response_dict = json.loads(raw_response)
            
            # Extract move and thoughts
            potential_move = response_dict.get("move", "")
            thoughts = response_dict.get("thoughts", "No thoughts provided")
            
            if not potential_move:
                raise ValueError("No move found in JSON response")
                
            # Validate the move
            move = self._extract_and_validate_move(potential_move, board)
            
            return move, thoughts
            
        except json.JSONDecodeError:
            # Fallback to old extraction method if not valid JSON
            move = self._extract_and_validate_move(raw_response, board)
            thoughts = raw_response
            return move, thoughts 