import os
import chess
from typing import Dict, Any
import asyncio
import logging

try:
    import anthropic
except ImportError:
    anthropic = None

from .base import ChessEngine
from core.types import MoveRequest, MoveResponse

logger = logging.getLogger(__name__)


class AnthropicEngine(ChessEngine):
    """Anthropic Claude chess engine implementation"""
    
    def __init__(self, model: str = "claude-4-sonnet-20250514", config: Dict[str, Any] = None):
        super().__init__(f"anthropic-{model}", config)
        self.model = model
        self.client = None
        self.available = False
    
    async def initialize(self) -> None:
        """Initialize the Anthropic client"""
        if anthropic is None:
            logger.error("Anthropic library not installed. Install with: pip install anthropic")
            return
            
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY environment variable not set")
            return
        
        try:
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
            # Test the connection
            await self._test_connection()
            self.available = True
            logger.info(f"Anthropic engine initialized successfully with model {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            self.available = False
    
    async def _test_connection(self) -> None:
        """Test the Anthropic API connection"""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "Test"}]
            )
            logger.info("Anthropic API connection test successful")
        except Exception as e:
            raise Exception(f"Anthropic API connection test failed: {e}")
    
    async def shutdown(self) -> None:
        """Clean up resources"""
        if self.client:
            await self.client.close()
    
    async def get_move(self, request: MoveRequest) -> MoveResponse:
        """Get a move from Claude"""
        if not self.available or not self.client:
            raise Exception("Anthropic engine not available. Check API key and connection.")
            
        try:
            # Create board from FEN
            board = chess.Board(request.fen)
            
            # Build prompt focusing on PGN moves
            prompt = self._build_prompt(board, request.pgn)
            
            logger.info(f"Sending request to Anthropic {self.model}")
            
            # Get response from Claude with timeout
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.model,
                    max_tokens=50,
                    temperature=min(max(request.temperature, 0.0), 1.0),
                    system="You are a chess grandmaster. Analyze the position and respond with only the best chess move in standard algebraic notation (e.g., e4, Nf3, O-O, Qxd5). Do not include explanations or commentary.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                ),
                timeout=35.0
            )
            
            raw_response = response.content[0].text.strip()
            logger.info(f"Anthropic raw response: {raw_response}")
            
            # Extract move from response
            move = self._extract_move(raw_response, board)
            
            return MoveResponse(
                move=move,
                raw_response=raw_response,
                thoughts=f"Claude {self.model}: {raw_response}"
            )
            
        except asyncio.TimeoutError:
            raise Exception("Anthropic API request timed out")
        except Exception as e:
            logger.error(f"Anthropic engine error: {e}")
            raise Exception(f"Anthropic engine error: {str(e)}")
    
    def _build_prompt(self, board: chess.Board, pgn_moves: list) -> str:
        """Build the prompt for Claude, emphasizing PGN context"""
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
    
    def _extract_move(self, response: str, board: chess.Board) -> str:
        """Extract a valid move from the response"""
        # Clean the response
        response = response.strip()
        
        # Try to find a valid move in the response
        potential_moves = []
        
        # Split by various delimiters
        for delimiter in [' ', '\n', '\t', ',', '.', '!', '?', ':', ';']:
            potential_moves.extend(response.split(delimiter))
        
        # Also try the whole response
        potential_moves.append(response)
        
        for potential_move in potential_moves:
            clean_move = potential_move.strip(".,!?()[]{}\"' \n\t")
            
            if not clean_move:
                continue
                
            try:
                # Try to parse as a move
                move = board.parse_san(clean_move)
                return clean_move
            except:
                continue
        
        # If no valid move found, return a random legal move as fallback
        legal_moves = list(board.legal_moves)
        if legal_moves:
            fallback_move = board.san(legal_moves[0])
            logger.warning(f"No valid move found in '{response}', using fallback: {fallback_move}")
            return fallback_move
        
        raise Exception("No legal moves available") 