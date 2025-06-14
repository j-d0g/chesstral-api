# ChessGPT API Architecture Guide

This is the FastAPI backend for ChessGPT, providing chess move generation, position evaluation, and engine management.

## 🏗️ Architecture Overview

```
main.py (FastAPI App)
├── /api/move (Move Generation)
├── /api/eval (Position Evaluation)  
├── /api/engines (Engine Listing)
├── /api/health (Health Check)
└── /api/rate_move (Move Rating)

engines/ (AI Engine Implementations)
├── base.py (Abstract Base Class)
├── engine_factory.py (Engine Management)
├── openai_engine.py (OpenAI GPT)
├── anthropic_engine.py (Claude)
├── gemini_engine.py (Google Gemini)
├── deepseek_engine.py (DeepSeek)
├── nanogpt_engine.py (NanoGPT)
└── stockfish_engine.py (Stockfish)

core/ (Type Definitions)
├── types.py (Pydantic Models)

service/ (Business Logic)
├── error_handler.py (Error Management)

util/ (Utilities)
repository/ (Data Access)
```

## 🎯 API Endpoints

### **POST /api/move** - Get AI Move
**Purpose**: Generate a chess move from an AI engine

**Request Model**: `MoveRequest`
```python
{
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "pgn": ["e4", "e5", "Nf3"],
    "engine": "openai",
    "model": "gpt-4",
    "temperature": 0.1
}
```

**Response Model**: `MoveResponse`
```python
{
    "move": "Nf3",
    "thoughts": "Developing the knight to control the center",
    "evaluation": 0.2,
    "raw_response": "..."
}
```

### **POST /api/eval** - Evaluate Position
**Purpose**: Get position evaluation from Stockfish

**Request Model**: `EvaluationRequest`
```python
{
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "depth": 15
}
```

### **GET /api/engines** - List Available Engines
**Purpose**: Get all available AI engines and their models

**Response**:
```python
{
    "engines": [
        {
            "type": "openai",
            "name": "OpenAI GPT",
            "models": ["gpt-3.5-turbo", "gpt-4", "o1"]
        }
    ]
}
```

### **POST /api/rate_move** - Rate Move Quality
**Purpose**: Store human ratings of AI moves for training data

## 🤖 Engine System

### **Base Engine Class** (`engines/base.py`)
```python
class ChessEngine(ABC):
    @abstractmethod
    async def get_move(self, request: MoveRequest) -> MoveResponse:
        """Generate a chess move for the given position"""
        pass
```

### **Adding a New Engine**

1. **Create Engine Class**:
```python
# engines/my_engine.py
from .base import ChessEngine
from core.types import MoveRequest, MoveResponse

class MyEngine(ChessEngine):
    def __init__(self, model: str):
        self.model = model
        # Initialize your engine here
    
    async def get_move(self, request: MoveRequest) -> MoveResponse:
        # Your move generation logic
        move = await self._generate_move(request.fen, request.pgn)
        
        return MoveResponse(
            move=move,
            thoughts="AI reasoning here",
            evaluation=0.0
        )
    
    async def _generate_move(self, fen: str, pgn: list) -> str:
        # Implement your specific AI logic
        pass
```

2. **Register in Factory** (`engines/engine_factory.py`):
```python
# Add to EngineType enum
class EngineType(str, Enum):
    MY_ENGINE = "my_engine"

# Add to create_engine method
elif engine_type == EngineType.MY_ENGINE:
    from .my_engine import MyEngine
    return MyEngine(model)
```

3. **Add to Frontend** (`web/src/components/EngineSelector.tsx`):
```typescript
const engines = [
    { 
        type: 'my_engine', 
        name: 'My Engine', 
        models: ['default', 'advanced'] 
    }
]
```

## 🔧 Core Types (`core/types.py`)

### **Request Models**
```python
class MoveRequest(BaseModel):
    fen: str                    # Current board position
    pgn: List[str]             # Move history
    engine: str                # Engine type
    model: Optional[str]       # Engine model
    temperature: float = 0.1   # AI creativity (0.0-1.0)

class EvaluationRequest(BaseModel):
    fen: str                   # Position to evaluate
    depth: Optional[int] = 15  # Search depth
```

### **Response Models**
```python
class MoveResponse(BaseModel):
    move: str                      # Move in SAN notation
    thoughts: Optional[str]        # AI reasoning
    evaluation: Optional[float]    # Position evaluation
    raw_response: Optional[str]    # Raw AI response

class ErrorResponse(BaseModel):
    error: str                     # Error message
    error_type: str               # Error category
    suggestions: List[str]        # Recovery suggestions
```

## 🛠️ Error Handling

The API includes comprehensive error handling with retry logic:

- **Illegal moves**: Automatic retry with legal move hints
- **API timeouts**: Exponential backoff retry
- **Invalid formats**: Move parsing with suggestions
- **Engine failures**: Graceful fallback options

## 🔍 Logging & Debugging

### **Log Levels**
- `DEBUG`: Detailed engine communication
- `INFO`: Request/response summaries  
- `WARNING`: Recoverable errors
- `ERROR`: Critical failures

### **Environment Variables**
```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=...

# Configuration
LOG_LEVEL=INFO
STOCKFISH_PATH=/usr/local/bin/stockfish
```

## 🚀 Adding New Features

### **New API Endpoint**
```python
@app.post("/api/my_feature")
async def my_feature(request: MyRequest) -> MyResponse:
    """
    PURPOSE: What this endpoint does
    """
    try:
        # Validate request
        # Process logic
        # Return response
        return MyResponse(...)
    except Exception as e:
        logger.error(f"Error in my_feature: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### **New Pydantic Model**
```python
class MyRequest(BaseModel):
    """Request model for my feature"""
    field1: str
    field2: Optional[int] = None
    
    class Config:
        # Add validation rules
        pass
```

## 🧪 Testing Patterns

### **Engine Testing**
```python
async def test_my_engine():
    engine = MyEngine("default")
    request = MoveRequest(
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        pgn=[],
        engine="my_engine",
        model="default"
    )
    
    response = await engine.get_move(request)
    assert response.move in legal_moves
```

### **API Testing**
```python
def test_move_endpoint():
    response = client.post("/api/move", json={
        "fen": "...",
        "engine": "my_engine"
    })
    assert response.status_code == 200
    assert "move" in response.json()
```

## 📊 Performance Considerations

- **Async/await** throughout for non-blocking I/O
- **Connection pooling** for external APIs
- **Caching** for repeated positions (future)
- **Rate limiting** for API protection (future)

## 🔐 Security

- **API key management** via environment variables
- **CORS configuration** for frontend access
- **Input validation** with Pydantic models
- **Error sanitization** to prevent information leakage

This architecture makes it easy for LLMs to:
- **Understand the API structure** and endpoints
- **Add new engines** following established patterns
- **Extend functionality** with new endpoints
- **Debug issues** with comprehensive logging
- **Follow security best practices** 