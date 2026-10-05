import os
import chess
from typing import Dict, Any
import logging
import aiohttp

from .base import ChessEngine
from core.types import MoveRequest, MoveResponse

logger = logging.getLogger(__name__)


class DeepSeekEngine(ChessEngine):
    """DeepSeek chess engine implementation"""
    
    def __init__(self, model: str = "deepseek-r1", config: Dict[str, Any] = None):
        super().__init__(f"deepseek-{model}", config)
        self.model = model
        self.api_key = None
        self.available = False
        self.base_url = "https://api.deepseek.com/v1"
    
    async def initialize(self) -> None:
        """Initialize the DeepSeek client"""
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            logger.error("DEEPSEEK_API_KEY environment variable not set")
            return
        
        try:
            # Test the connection
            await self._test_connection()
            self.available = True
            logger.info(f"DeepSeek engine initialized successfully with model {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek client: {e}")
            self.available = False
    
    async def _test_connection(self) -> None:
        """Test the DeepSeek API connection"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Test"}],
                    "max_tokens": 1
                }
                
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info("DeepSeek API connection test successful")
                    else:
                        raise Exception(f"API test failed with status {response.status}")
        except Exception as e:
            raise Exception(f"DeepSeek API connection test failed: {e}")
    
    async def shutdown(self) -> None:
        """Clean up resources"""
        # No explicit cleanup needed for HTTP client
        pass
    
    async def get_move(self, request: MoveRequest) -> MoveResponse:
        """Get a move from DeepSeek with proper reprompting for invalid moves"""
        if not self.available or not self.api_key:
            raise Exception("DeepSeek engine not available. Check API key and connection.")
            
        try:
            # Create board from FEN
            board = chess.Board(request.fen)
            
            # Use reprompting loop for robust move generation
            move, raw_response, thoughts = await self.reprompt_loop(board, request)
            
            return MoveResponse(
                move=move,
                raw_response=raw_response,
                thoughts=thoughts
            )
            
        except Exception as e:
            logger.error(f"DeepSeek engine error: {e}")
            raise Exception(f"DeepSeek engine error: {str(e)}")
    
    async def _make_api_call(self, prompt: str, system_msg: str, temperature: float) -> str:
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": 50,
            "stream": False
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=35)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"DeepSeek API error {response.status}: {error_text}")

                result = await response.json()

        return result["choices"][0]["message"]["content"].strip()