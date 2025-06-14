# ChessGPT API

This directory contains the FastAPI-based Python backend for the ChessGPT application. It serves as the bridge between the frontend and various chess engines, including language model-based engines like NanoGPT.

## Key Features

-   **Multiple Engine Support:** Provides a unified interface for different chess engines (NanoGPT, Stockfish, OpenAI, etc.).
-   **FastAPI Framework:** Built on FastAPI for high performance and automatic interactive documentation (via Swagger UI).
-   **Detailed Logging:** Configured for comprehensive logging, crucial for debugging and ML research.

## NanoGPT Engine (`nanogpt_engine.py`)

The backend includes a highly customized engine for interfacing with our fine-tuned NanoGPT model. This was the focus of a significant debugging and refactoring effort.

### PGN History and Prompt Formatting

The NanoGPT model requires the full game history in PGN format to make accurate, legal moves. A critical component of the engine is its ability to correctly format this history into a prompt.

-   **Input:** The engine receives a list of moves from the frontend (e.g., `['e4', 'c5', 'Nf3']`).
-   **Processing:** It formats this list into a standard PGN string: `1. e4 c5 2. Nf3`.
-   **Prompting:** A special semicolon delimiter is prepended to the string (`';1. e4 c5 2. Nf3'`) before it's sent to the model, as this matches the format used during training.

### Research-Oriented Debugging

To aid in ML research and diagnose model behavior, the NanoGPT engine's `get_move` method was enhanced with extensive logging:

-   **Verbose Output:** The console now prints the exact PGN prompt sent to the model, the raw output, the token-by-token generation process, and the temperature scaling for each retry attempt.
-   **Illegal Move Handling:** The temperature scaling logic was corrected to match the original research script, starting low (0.001) and gradually increasing upon illegal move attempts. This ensures the model first tries to find the most probable move before exploring more creative (and potentially risky) options.

## Endpoints

The main endpoints are defined in `src/main.py`:

-   `/api/move`: Receives a game state (FEN, PGN) and returns an AI-generated move.
-   `/api/eval`: Evaluates the current position using Stockfish.
-   `/api/engines`: Returns a list of available chess engines.

## Running the API

To run the backend server:

```bash
cd chess_viz/api
uvicorn src.main:app --reload
```

The API will be available at `http://localhost:8000`.
