import os
import chess
from typing import Dict, Any
import asyncio
import logging
from openai import AsyncOpenAI

from .base import ChessEngine
from core.types import MoveRequest, MoveResponse


logger = logging.getLogger(__name__)


class OpenAIEngine(ChessEngine):
    """OpenAI GPT chess engine implementation"""
    
    def __init__(self, model: str = "o1-mini", config: Dict[str, Any] = None):
        super().__init__(f"openai-{model}", config)
        self.model = model
        self.client: AsyncOpenAI = None
        self.available = False
    
    async def initialize(self) -> None:
        """Initialize OpenAI client and test connection"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OpenAI API key not found. OpenAI engine will not be available.")
            return
            
        try:
            self.client = AsyncOpenAI(api_key=api_key)
            await self._test_connection()
            self.available = True
            logger.info(f"✅ OpenAI {self.model} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI {self.model}: {e}")
            self.available = False
    
    async def _test_connection(self) -> None:
        """Test OpenAI API connection"""
        try:
            if self.model.startswith('o1'):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": "Test"}
                    ],
                    timeout=30.0
                )
            else:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Test"},
                        {"role": "user", "content": "Test"}
                    ],
                    temperature=0.1,
                    max_tokens=1,
                    timeout=30.0
                )
            logger.info("OpenAI API connection test successful")
        except Exception as e:
            raise Exception(f"OpenAI API connection test failed: {e}")
    
    async def shutdown(self) -> None:
        """Clean up resources"""
        if self.client:
            await self.client.close()
    
    async def get_move(self, request: MoveRequest) -> MoveResponse:
        """Get a move from OpenAI using the shared reprompt loop"""
        if not self.available or not self.client:
            raise Exception("OpenAI engine not available. Check API key and connection.")
            
        try:
            # Create board from FEN
            board = chess.Board(request.fen)
            
            # Use shared reprompt loop from base class
            move, raw_response, thoughts = await self.reprompt_loop(board, request)
            
            return MoveResponse(
                move=move,
                raw_response=raw_response,
                thoughts=thoughts
            )
            
        except Exception as e:
            logger.error(f"OpenAI engine error: {e}")
            raise Exception(f"OpenAI engine error: {str(e)}")
    
    async def _make_api_call(self, prompt: str, system_msg: str, temperature: float) -> str:
        """Make API call to OpenAI - implements abstract method from base class"""
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
                    temperature=temperature,
                    max_tokens=50,
                    timeout=30.0
                ),
                timeout=35.0
            )
        
        return response.choices[0].message.content.strip()
    
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