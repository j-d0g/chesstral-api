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
            move, raw_response, thoughts = await self.reprompt_loop(board, request)
            
            return MoveResponse(
                move=move,
                raw_response=raw_response,
                thoughts=thoughts
            )
            
        except Exception as e:
            logger.error(f"Gemini engine error: {e}")
            raise Exception(f"Gemini engine error: {str(e)}")
    
    async def _make_api_call(self, prompt: str, system_msg: str, temperature: float) -> str:
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=50,
            temperature=temperature,
            candidate_count=1
        )

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

        try:
            raw_response = response.text.strip()
        except ValueError:
            raw_response = ""
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    raw_response += part.text
            raw_response = raw_response.strip()

        return raw_response

    def build_prompt(self, board: chess.Board, pgn_moves: list) -> str:
        return super().build_prompt(board, pgn_moves) + " Respond with only the move in standard algebraic notation (e.g., e4, Nf3, O-O, Qxd5)."