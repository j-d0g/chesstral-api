from abc import ABC, abstractmethod
from typing import Dict, Any
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import MoveRequest, MoveResponse, EvaluationRequest, EvaluationResponse


class ChessEngine(ABC):
    """Abstract base class for all chess engines"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
    
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
    
    async def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        """Evaluate a position (optional, not all engines support this)"""
        raise NotImplementedError(f"{self.name} does not support position evaluation") 