import sys
import os
# Add the src directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import traceback
from typing import Dict
import chess

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Check multiple possible locations for .env file
    env_paths = [
        ".env",  # Current directory (when running from api/)
        "../.env",  # Parent directory (when running from src/)
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")  # Absolute path to api/.env
    ]
    
    env_loaded = False
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path)
            print(f"✅ Environment variables loaded from {os.path.abspath(env_path)}")
            env_loaded = True
            break
    
    if not env_loaded:
        print("⚠️  No .env file found in expected locations")
        
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("⚠️  Loading environment variables from system environment only")

from core.types import (
    MoveRequest, MoveResponse, MoveRatingRequest, 
    EvaluationRequest, EvaluationResponse, ErrorResponse
)
from engines.engine_factory import EngineFactory
from engines.catalog import ENGINE_CATALOG
from service.ratings import save_rating

# Set up logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log API key status (without revealing the keys)
def log_api_key_status():
    """Log which API keys are available without revealing their values"""
    api_keys = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'), 
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),

        'DEEPSEEK_API_KEY': os.getenv('DEEPSEEK_API_KEY')
    }
    
    logger.info("=== API Key Status ===")
    for key_name, key_value in api_keys.items():
        if key_value:
            masked_key = key_value[:8] + "..." + key_value[-4:] if len(key_value) > 12 else "***"
            logger.info(f"✅ {key_name}: {masked_key}")
        else:
            logger.warning(f"❌ {key_name}: Not set")
    logger.info("=====================")

# Log API key status on startup
log_api_key_status()

# Global engine factory
engine_factory: EngineFactory = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global engine_factory
    
    # Startup
    logger.info("Starting ChessGPT API...")
    try:
        engine_factory = EngineFactory()
        await engine_factory.initialize()
        logger.info("Engine factory initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize engine factory: {e}")
        logger.error(traceback.format_exc())
        raise
    
    yield
    
    # Shutdown  
    logger.info("Shutting down ChessGPT API...")
    if engine_factory:
        try:
            await engine_factory.shutdown()
            logger.info("Engine factory shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="ChessGPT API",
    description="Chess API with multiple AI engines",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/move", response_model=MoveResponse)
async def get_computer_move(request: MoveRequest) -> MoveResponse:
    """Get a move from the specified engine with comprehensive error handling"""
    from service.error_handler import (
        MoveErrorTracker, ChessErrorHandler, log_move_attempt
    )
    
    # Initialize error tracking
    error_tracker = MoveErrorTracker(max_retries=5)
    
    try:
        logger.info("=" * 60)
        logger.info(f"🎯 MOVE REQUEST RECEIVED")
        logger.info(f"   Engine: {request.engine}")
        logger.info(f"   Model: {request.model}")
        logger.info(f"   FEN: {request.fen}")
        logger.info(f"   PGN moves: {request.pgn}")
        logger.info(f"   Temperature: {request.temperature}")
        logger.info("=" * 60)
        
        # Validate request data
        if not request.fen:
            logger.error("❌ FEN is empty or None")
            move_error = ChessErrorHandler.create_move_error(
                error_type="missing_data",
                error_message="FEN position is required",
                engine_type=request.engine
            )
            raise HTTPException(status_code=400, detail={
                "error": "FEN position is required",
                "error_type": "missing_data",
                "move_error": move_error.dict()
            })
        
        if not request.engine:
            logger.error("❌ Engine is empty or None")
            move_error = ChessErrorHandler.create_move_error(
                error_type="missing_data",
                error_message="Engine type is required",
                engine_type="unknown"
            )
            raise HTTPException(status_code=400, detail={
                "error": "Engine type is required",
                "error_type": "missing_data",
                "move_error": move_error.dict()
            })
        
        if not request.model:
            logger.error("❌ Model is empty or None")
            move_error = ChessErrorHandler.create_move_error(
                error_type="missing_data",
                error_message="Model is required",
                engine_type=request.engine
            )
            raise HTTPException(status_code=400, detail={
                "error": "Model is required",
                "error_type": "missing_data",
                "move_error": move_error.dict()
            })
        
        # Validate FEN format
        try:
            board = chess.Board(request.fen)
            logger.info(f"✅ FEN validation passed - {board.turn} to move")
            logger.info(f"🎲 Legal moves: {[board.san(move) for move in list(board.legal_moves)]}")
        except Exception as e:
            logger.error(f"❌ Invalid FEN format: {e}")
            move_error = ChessErrorHandler.create_move_error(
                error_type="invalid_position",
                error_message=f"The chess position is invalid: {str(e)}",
                engine_type=request.engine
            )
            error_response = ChessErrorHandler.create_error_response(
                error=e,
                error_context="FEN validation failed",
                move_error=move_error
            )
            raise HTTPException(status_code=400, detail=error_response.dict())
        
        # Check if game is over
        if board.is_game_over():
            error_msg = f"Game is already over. Result: {board.result()}"
            logger.error(f"❌ {error_msg}")
            move_error = ChessErrorHandler.create_move_error(
                error_type="game_over",
                error_message=f"The game has ended: {board.result()}",
                board=board,
                engine_type=request.engine
            )
            raise HTTPException(status_code=400, detail={
                "error": error_msg,
                "error_type": "game_over",
                "move_error": move_error.dict()
            })
        
        if not engine_factory:
            logger.error("❌ Engine factory not initialized")
            move_error = ChessErrorHandler.create_move_error(
                error_type="system_error",
                error_message="Chess engine system is not initialized",
                engine_type=request.engine
            )
            raise HTTPException(status_code=503, detail={
                "error": "Engine factory not initialized",
                "error_type": "system_error",
                "move_error": move_error.dict()
            })
        
        logger.info(f"🔍 Checking engine availability...")
        
        # Get current availability status
        available_engines = engine_factory.get_available_engines()
        logger.info(f"📊 Engine availability: {available_engines}")
        
        # Validate engine availability
        if not engine_factory.is_engine_available(request.engine):
            available_list = [name for name, available in available_engines.items() if available]
            error_msg = f"Engine '{request.engine}' is not available"
            logger.error(f"❌ {error_msg}")
            
            move_error = ChessErrorHandler.create_move_error(
                error_type="engine_unavailable",
                error_message=f"The {request.engine} engine is currently unavailable. Available engines: {available_list}",
                engine_type=request.engine
            )
            
            raise HTTPException(status_code=400, detail={
                "error": error_msg,
                "error_type": "engine_unavailable",
                "move_error": move_error.dict(),
                "suggestions": [
                    f"Try one of these available engines: {', '.join(available_list)}",
                    "For frontier models, ensure API keys are set in .env file",
                    "Check the engine configuration"
                ],
                "recovery_options": [
                    "Switch to NanoGPT (always available)",
                    "Verify API keys in .env file",
                    "Contact support if the issue persists"
                ]
            })
        
        logger.info(f"✅ Engine '{request.engine}' is available")
        
        # Get the engine instance
        logger.info(f"🔧 Creating engine instance...")
        try:
            engine = await engine_factory.get_engine(request.engine, request.model)
            if not engine:
                error_msg = f"Could not create engine '{request.engine}' with model '{request.model}'"
                logger.error(f"❌ {error_msg}")
                
                move_error = ChessErrorHandler.create_move_error(
                    error_type="engine_creation_failed",
                    error_message=f"Failed to create {request.engine} engine with model {request.model}",
                    engine_type=request.engine
                )
                
                raise HTTPException(status_code=404, detail={
                    "error": error_msg,
                    "error_type": "engine_creation_failed",
                    "move_error": move_error.dict()
                })
            
            logger.info(f"✅ Engine instance created: {engine.name}")
            
        except Exception as e:
            error_msg = f"Failed to create engine '{request.engine}'"
            logger.error(f"❌ {error_msg}: {str(e)}")
            logger.error(f"🔍 Exception details: {traceback.format_exc()}")
            
            move_error = ChessErrorHandler.create_move_error(
                error_type="engine_creation_failed",
                error_message=f"Could not initialize the {request.engine} engine: {str(e)}",
                engine_type=request.engine
            )
            
            error_response = ChessErrorHandler.create_error_response(
                error=e,
                error_context="Engine creation failed",
                move_error=move_error
            )
            
            raise HTTPException(status_code=500, detail=error_response.dict())
        
        # Make the move request with comprehensive error tracking
        logger.info(f"🚀 Requesting move from engine...")
        try:
            response = await engine.get_move(request)
            logger.info(f"✅ Move received: {response.move}")
            logger.info(f"💭 Engine thoughts: {response.thoughts}")
            logger.info(f"📝 Raw response: {response.raw_response}")
            
            # CRITICAL: Final validation to ensure NO invalid moves leave the backend
            if response.move:
                try:
                    # Validate the move against the current position
                    validation_board = chess.Board(request.fen)
                    parsed_move = validation_board.parse_san(response.move.strip())
                    if parsed_move not in validation_board.legal_moves:
                        raise chess.IllegalMoveError(f"Move {response.move} is not legal in position {request.fen}")
                    logger.info(f"🔒 Final validation passed: {response.move} is legal")
                    
                    # Add success metrics to response
                    response.response_time_ms = error_tracker.get_response_time_ms()
                    response.engine_status = "success"
                    
                except Exception as validation_error:
                    error_msg = f"CRITICAL: Engine returned invalid move '{response.move}'"
                    logger.error(f"❌ {error_msg}: {validation_error}")
                    
                    # Track this critical error
                    error_tracker.add_attempt(
                        attempt_number=1,
                        attempted_move=response.move,
                        error_type="illegal_move",
                        error_message=str(validation_error),
                        raw_response=response.raw_response,
                        board=validation_board
                    )
                    
                    move_error = ChessErrorHandler.create_move_error(
                        error_type="illegal_move",
                        error_message=f"The AI suggested an illegal move: {response.move}",
                        attempted_move=response.move,
                        board=validation_board,
                        total_attempts=1,
                        failed_attempts=error_tracker.get_failed_attempts(),
                        engine_type=request.engine
                    )
                    
                    error_response = ChessErrorHandler.create_error_response(
                        error=validation_error,
                        error_context="Final move validation failed",
                        move_error=move_error
                    )
                    
                    raise HTTPException(status_code=500, detail=error_response.dict())
            else:
                error_msg = "CRITICAL: Engine returned empty or null move"
                logger.error(f"❌ {error_msg}")
                
                move_error = ChessErrorHandler.create_move_error(
                    error_type="no_move_generated",
                    error_message="The AI engine did not generate any move",
                    board=board,
                    engine_type=request.engine
                )
                
                raise HTTPException(status_code=500, detail={
                    "error": error_msg,
                    "error_type": "no_move_generated",
                    "move_error": move_error.dict()
                })
            
            # Log successful move generation
            log_move_attempt(error_tracker, f"{request.engine} ({request.model})" if request.model else request.engine, board)
            
            logger.info("=" * 60)
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            error_msg = f"Engine '{request.engine}' failed to generate move"
            logger.error(f"❌ {error_msg}: {str(e)}")
            logger.error(f"🔍 Exception details: {traceback.format_exc()}")
            
            # Classify the error type
            error_type, classified_message = ChessErrorHandler.classify_move_error(e, "", board)
            
            move_error = ChessErrorHandler.create_move_error(
                error_type=error_type,
                error_message=classified_message,
                board=board,
                total_attempts=1,
                engine_type=request.engine
            )
            
            error_response = ChessErrorHandler.create_error_response(
                error=e,
                error_context=f"Move generation failed for {request.engine}",
                move_error=move_error
            )
            
            raise HTTPException(status_code=500, detail=error_response.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error in move request"
        logger.error(f"❌ {error_msg}: {str(e)}")
        logger.error(f"🔍 Exception details: {traceback.format_exc()}")
        
        error_response = ChessErrorHandler.create_error_response(
            error=e,
            error_context="Unexpected system error"
        )
        
        raise HTTPException(status_code=500, detail=error_response.dict())


@app.post("/api/rate_move")
async def save_move_rating(request: MoveRatingRequest) -> Dict[str, str]:
    try:
        chess.Board(request.fen)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid FEN") from exc

    save_rating(request)
    return {"status": "saved"}


@app.post("/api/eval", response_model=EvaluationResponse)
async def eval_board(request: EvaluationRequest) -> EvaluationResponse:
    """Evaluate a board position using Stockfish"""
    try:
        logger.info(f"Evaluation request for FEN: {request.fen[:50]}...")
        
        if not engine_factory:
            raise HTTPException(status_code=503, detail="Engine factory not initialized")
        
        engine = await engine_factory.get_engine("stockfish")
        if not engine:
            raise HTTPException(status_code=404, detail="Stockfish engine not available")
        
        response = await engine.evaluate(request)
        logger.info(f"Evaluation response: evaluation={response.evaluation}, best_move={response.best_move}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error evaluating position: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/api/engines")
async def list_engines() -> Dict[str, list]:
    """List available engines with their current status"""
    if not engine_factory:
        return {"engines": []}
    
    from engines.nanogpt_paths import available_models

    availability = engine_factory.get_available_engines()
    engines_config = []
    for engine_type, definition in ENGINE_CATALOG.items():
        available = availability.get(engine_type, False)
        engines_config.append(
            {
                "name": definition.name,
                "models": (
                    available_models()
                    if engine_type == "nanogpt"
                    else list(definition.models)
                ),
                "description": definition.description,
                "status": "available" if available else "unavailable",
                "reason": engine_factory.get_engine_reason(engine_type),
            }
        )

    return {"engines": engines_config}


@app.get("/api/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint with detailed engine status"""
    try:
        status = {"status": "healthy", "version": "2.0.0"}
        
        if engine_factory:
            availability = engine_factory.get_available_engines()
            available_count = sum(1 for available in availability.values() if available)
            total_count = len(availability)
            
            status.update({
                "engines_available": f"{available_count}/{total_count}",
                "engine_details": str(availability)
            })
        else:
            status["engines_available"] = "0/0 (factory not initialized)"
            status["engine_details"] = "Engine factory not initialized"
        
        return status
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "unhealthy", "error": str(e), "engine_details": "Error occurred"}


@app.get("/api/keys/status")
async def get_api_keys_status():
    """Get the status of API keys (whether they are set or not)"""
    return {
        'OPENAI_API_KEY': bool(os.getenv('OPENAI_API_KEY')),
        'ANTHROPIC_API_KEY': bool(os.getenv('ANTHROPIC_API_KEY')),
        'GOOGLE_API_KEY': bool(os.getenv('GOOGLE_API_KEY')),
        'DEEPSEEK_API_KEY': bool(os.getenv('DEEPSEEK_API_KEY')),
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 