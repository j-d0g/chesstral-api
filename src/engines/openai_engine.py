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