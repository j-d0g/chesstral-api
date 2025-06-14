import os
import chess
from typing import Dict, Any
import asyncio
import logging

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from .base import ChessEngine
from core.types import MoveRequest, MoveResponse

logger = logging.getLogger(__name__)


class GeminiEngine(ChessEngine):
    """Google Gemini chess engine implementation"""
    
    def __init__(self, model: str = "gemini-2.0-flash", config: Dict[str, Any] = None):
        super().__init__(f"gemini-{model}", config)
        self.model = model
        self.client = None
        self.available = False
    
    async def initialize(self) -> None:
        """Initialize the Gemini client"""
        if genai is None:
            logger.error("Google Generative AI library not installed. Install with: pip install google-generativeai")
            return
            
        try:
            # Get API key from environment
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.error("GOOGLE_API_KEY environment variable not set")
                raise ValueError("Google API key not configured")
            
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model)
            # Test the connection
            await self._test_connection()
            self.available = True
            logger.info(f"Gemini engine initialized successfully with model {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.available = False
    
    async def _test_connection(self) -> None:
        """Test the Gemini API connection"""
        try:
            response = await asyncio.to_thread(
                self.client.generate_content,
                "Test",
                generation_config=genai.types.GenerationConfig(max_output_tokens=1)
            )
            logger.info("Gemini API connection test successful")
        except Exception as e:
            raise Exception(f"Gemini API connection test failed: {e}")
    
    async def shutdown(self) -> None:
        """Clean up resources"""
        # Gemini client doesn't need explicit cleanup
        pass
    
    async def get_move(self, request: MoveRequest) -> MoveResponse:
        """Get a move from Gemini with proper reprompting for invalid moves"""
        if not self.available or not self.client:
            raise Exception("Gemini engine not available. Check API key and connection.")
            
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
            logger.error(f"Gemini engine error: {e}")
            raise Exception(f"Gemini engine error: {str(e)}")
    
    async def _reprompt_loop(self, board: chess.Board, request: MoveRequest, max_retries: int = 5) -> tuple[str, str, str]:
        """Reprompting loop with move validation and specific error feedback"""
        
        for attempt in range(max_retries):
            try:
                # Build prompt for this attempt
                if attempt == 0:
                    prompt = self._build_prompt(board, request.pgn)
                else:
                    # For retries, include the error message in the prompt
                    prompt = f"{self._last_error_message}\n\nOriginal position:\n{self._build_prompt(board, request.pgn)}"
                
                logger.info(f"Gemini attempt {attempt + 1}/{max_retries}")
                
                # Use shared temperature calculation from base class
                eval_temperature = self.calculate_retry_temperature(request.temperature, attempt, max_retries)
                
                # Configure generation parameters
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=50,
                    temperature=eval_temperature,
                    candidate_count=1
                )
                
                # Get response from Gemini with timeout
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.generate_content,
                        prompt,
                        generation_config=generation_config,
                        safety_settings=[
                            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        ]
                    ),
                    timeout=35.0
                )
                
                # Handle both simple text and multi-part responses
                try:
                    raw_response = response.text.strip()
                except ValueError:
                    # For multi-part responses, extract text from parts
                    raw_response = ""
                    if hasattr(response, 'candidates') and response.candidates:
                        for candidate in response.candidates:
                            if hasattr(candidate, 'content') and candidate.content:
                                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text') and part.text:
                                            raw_response += part.text
                    raw_response = raw_response.strip()
                
                logger.info(f"Gemini raw response: '{raw_response}'")
                
                # Extract and validate move
                try:
                    move = self._extract_and_validate_move(raw_response, board)
                    # Success!
                    thoughts = f"Gemini {self.model} (attempt {attempt + 1}): {raw_response}"
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
                    raise Exception("Gemini API request timed out after multiple attempts")
                continue
                
        raise Exception(f"Failed to get valid move after {max_retries} attempts")
    
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
        
        # If no valid move found, provide helpful error message
        raise ValueError(f"No valid chess move found in response: '{response}'. Legal moves include: {', '.join(legal_moves_san[:10])}...")
    
    def _build_prompt(self, board: chess.Board, pgn_moves: list) -> str:
        """Build the prompt for Gemini, emphasizing PGN context"""
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
        prompt_parts.append(f"What is the best move for {color}? Respond with only the move in standard algebraic notation (e.g., e4, Nf3, O-O, Qxd5).")
        
        return "\n".join(prompt_parts) 