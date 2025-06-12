from typing import Dict, Optional
import logging
import traceback

from .base import ChessEngine
from core.types import EngineType

logger = logging.getLogger(__name__)


class EngineFactory:
    """Factory for creating and managing chess engines"""
    
    def __init__(self):
        self._engines: Dict[str, ChessEngine] = {}
        self._initialized = False
        self._engine_availability: Dict[str, bool] = {}
    
    async def initialize(self):
        """Initialize the factory and test engine availability"""
        if self._initialized:
            return
            
        logger.info("Initializing engine factory...")
        
        # Test availability of all engines
        await self._test_engine_availability()
        
        self._initialized = True
        logger.info("Engine factory initialized")
    
    async def _test_engine_availability(self):
        """Test which engines are available"""
        logger.info("🔍 Testing engine availability...")
        
        engine_tests = [
            ("nanogpt", "small-16"),
            ("stockfish", None),
            ("openai", "o1-mini"),
            ("anthropic", "claude-4-sonnet-20250514"),
            ("gemini", "gemini-2.0-flash"),
            ("deepseek", "deepseek-chat"),
        ]
        
        for engine_type, model in engine_tests:
            logger.info(f"🧪 Testing {engine_type} engine...")
            try:
                # Create engine instance
                logger.info(f"   Creating {engine_type} engine instance...")
                engine = await self._create_engine(engine_type, model)
                
                # Initialize engine
                logger.info(f"   Initializing {engine_type} engine...")
                await engine.initialize()
                
                # Check if engine is actually available
                available = getattr(engine, 'available', True)
                self._engine_availability[engine_type] = available
                
                if available:
                    logger.info(f"✅ {engine_type} engine available and ready")
                else:
                    logger.warning(f"❌ {engine_type} engine created but not available (likely missing API key)")
                
                # Clean up
                logger.info(f"   Shutting down {engine_type} test instance...")
                await engine.shutdown()
                
            except ImportError as e:
                logger.warning(f"❌ {engine_type} engine unavailable - missing dependency: {e}")
                self._engine_availability[engine_type] = False
            except Exception as e:
                logger.warning(f"❌ {engine_type} engine unavailable - error: {e}")
                logger.debug(f"   Full error details: {traceback.format_exc()}")
                self._engine_availability[engine_type] = False
        
        # Summary
        available_count = sum(1 for available in self._engine_availability.values() if available)
        total_count = len(self._engine_availability)
        logger.info(f"📊 Engine availability summary: {available_count}/{total_count} engines available")
        
        for engine_name, available in self._engine_availability.items():
            status = "✅ Available" if available else "❌ Unavailable"
            logger.info(f"   {engine_name}: {status}")
    
    def is_engine_available(self, engine_type: str) -> bool:
        """Check if an engine type is available"""
        return self._engine_availability.get(engine_type, False)
    
    async def get_engine(self, engine_type: str, model: Optional[str] = None) -> ChessEngine:
        """Get or create an engine instance"""
        # Check availability first
        if not self.is_engine_available(engine_type):
            raise ValueError(f"Engine '{engine_type}' is not available. Check API keys and dependencies.")
        
        engine_key = f"{engine_type}"
        if model:
            engine_key += f":{model}"
        
        if engine_key not in self._engines:
            engine = await self._create_engine(engine_type, model)
            await engine.initialize()
            self._engines[engine_key] = engine
            
        return self._engines[engine_key]
    
    async def _create_engine(self, engine_type: str, model: Optional[str] = None) -> ChessEngine:
        """Create a new engine instance"""
        
        if engine_type == "openai":
            from .openai_engine import OpenAIEngine
            return OpenAIEngine(model or "o1-mini")
            
        elif engine_type == "anthropic":
            from .anthropic_engine import AnthropicEngine
            return AnthropicEngine(model or "claude-4-sonnet-20250514")
            
        elif engine_type == "gemini":
            from .gemini_engine import GeminiEngine
            return GeminiEngine(model or "gemini-2.0-flash")
            
        elif engine_type == "deepseek":
            from .deepseek_engine import DeepSeekEngine
            return DeepSeekEngine(model or "deepseek-chat")
            
        elif engine_type == "replicate":
            from .replicate_engine import ReplicateEngine
            return ReplicateEngine(model or "meta/llama-2-70b-chat")
            
        elif engine_type == "stockfish":
            from .stockfish_engine import StockfishEngine
            return StockfishEngine()
            
        elif engine_type == "nanogpt":
            from .nanogpt_engine import NanoGPTEngine
            return NanoGPTEngine(model or "small-16")
            
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")
    
    def get_available_engines(self) -> Dict[str, bool]:
        """Get the availability status of all engines"""
        return self._engine_availability.copy()
    
    async def shutdown(self):
        """Shutdown all engines"""
        logger.info("Shutting down engines...")
        
        for engine in self._engines.values():
            try:
                await engine.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down engine {engine.name}: {e}")
        
        self._engines.clear()
        self._initialized = False
        logger.info("All engines shut down") 