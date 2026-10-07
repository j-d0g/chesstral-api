import os
from typing import Dict, Optional
import logging
import traceback

from .base import ChessEngine
from .catalog import ENGINE_CATALOG

logger = logging.getLogger(__name__)


class EngineFactory:
    """Factory for creating and managing chess engines"""
    
    def __init__(self):
        self._engines: Dict[str, ChessEngine] = {}
        self._initialized = False
        self._engine_availability: Dict[str, bool] = {}
        self._engine_reasons: Dict[str, str] = {}
    
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

        from .nanogpt_paths import available_models

        for engine_type, definition in ENGINE_CATALOG.items():
            logger.info(f"🧪 Testing {engine_type} engine...")
            if engine_type == "nanogpt":
                models = available_models()
                self._engine_availability[engine_type] = bool(models)
                self._engine_reasons[engine_type] = (
                    "Ready"
                    if models
                    else "No checkpoints in models/checkpoints (see models/README.md)"
                )
                continue

            if definition.required_env and not os.getenv(definition.required_env):
                self._engine_availability[engine_type] = False
                self._engine_reasons[engine_type] = f"Requires {definition.required_env}"
                continue

            model = definition.models[0] if definition.models else None
            try:
                # Create engine instance
                logger.info(f"   Creating {engine_type} engine instance...")
                engine = await self._create_engine(engine_type, model)
                
                # Initialize engine
                logger.info(f"   Initializing {engine_type} engine...")
                await engine.initialize()
                
                # Check if engine is actually available
                available = getattr(engine, "available", True)
                self._engine_availability[engine_type] = available
                self._engine_reasons[engine_type] = (
                    "Ready"
                    if available
                    else (
                        "Stockfish binary not found"
                        if engine_type == "stockfish"
                        else f"{engine_type.capitalize()} engine is unavailable"
                    )
                )
                
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
                self._engine_reasons[engine_type] = f"Missing dependency: {e}"
            except Exception as e:
                logger.warning(f"❌ {engine_type} engine unavailable - error: {e}")
                logger.debug(f"   Full error details: {traceback.format_exc()}")
                self._engine_availability[engine_type] = False
                self._engine_reasons[engine_type] = (
                    "Stockfish binary not found"
                    if engine_type == "stockfish"
                    else f"{engine_type.capitalize()} engine unavailable: {e}"
                )
        
        # Summary
        available_count = sum(1 for available in self._engine_availability.values() if available)
        total_count = len(self._engine_availability)
        logger.info(f"📊 Engine availability summary: {available_count}/{total_count} engines available")
        
        for engine_name, available in self._engine_availability.items():
            status = "✅ Available" if available else "❌ Unavailable"
            logger.info(f"   {engine_name}: {status}")
    
    def is_engine_available(self, engine_type: str) -> bool:
        """Check if an engine type is available"""
        if engine_type == "nanogpt":
            from .nanogpt_paths import available_models

            return bool(available_models())
        return self._engine_availability.get(engine_type, False)

    def get_engine_reason(self, engine_type: str) -> str:
        """Get a helpful availability message for an engine"""
        if engine_type == "nanogpt":
            return (
                "Ready"
                if self.is_engine_available(engine_type)
                else "No checkpoints in models/checkpoints (see models/README.md)"
            )
        return self._engine_reasons.get(engine_type, "Engine status is unknown")
    
    async def get_engine(self, engine_type: str, model: Optional[str] = None) -> ChessEngine:
        """Get or create an engine instance"""
        if engine_type == "nanogpt":
            from .nanogpt_paths import available_models, checkpoint_path

            model = model or next(iter(available_models()), None)
            if not model:
                raise ValueError(
                    "No NanoGPT checkpoints are available in models/checkpoints. "
                    "See models/README.md to download one."
                )
            if checkpoint_path(model) is None:
                raise ValueError(
                    f"NanoGPT checkpoint for '{model}' was not found in "
                    "models/checkpoints or the legacy models directory."
                )

        # Check availability first
        if not self.is_engine_available(engine_type):
            raise ValueError(
                f"Engine '{engine_type}' is not available: {self.get_engine_reason(engine_type)}"
            )
        
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

        definition = ENGINE_CATALOG.get(engine_type)
        if model is None and definition and definition.models:
            model = definition.models[0]

        if engine_type == "openai":
            from .openai_engine import OpenAIEngine
            return OpenAIEngine(model)
            
        elif engine_type == "anthropic":
            from .anthropic_engine import AnthropicEngine
            return AnthropicEngine(model or "claude-sonnet-4-20250514")
            
        elif engine_type == "gemini":
            from .gemini_engine import GeminiEngine
            return GeminiEngine(model)
            
        elif engine_type == "deepseek":
            from .deepseek_engine import DeepSeekEngine
            return DeepSeekEngine(model)
            
        elif engine_type == "stockfish":
            from .stockfish_engine import StockfishEngine
            return StockfishEngine()
            
        elif engine_type == "nanogpt":
            from .nanogpt_engine import NanoGPTEngine
            return NanoGPTEngine(model)
            
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")
    
    def get_available_engines(self) -> Dict[str, bool]:
        """Get the availability status of all engines"""
        availability = self._engine_availability.copy()
        if "nanogpt" in availability:
            availability["nanogpt"] = self.is_engine_available("nanogpt")
        return availability
    
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