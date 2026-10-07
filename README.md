# ChessGPT API

FastAPI service for the ChessGPT web app and its chess engines.

## Run

From the repository root:

```bash
.venv/bin/python -m uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

## Test

```bash
.venv/bin/python -m pytest -q
```

## NanoGPT checkpoints

Set `HF_TOKEN` in the environment or repository `.env`, then download a model:

```bash
.venv/bin/python models/download_checkpoint.py --model small-8
```

The default location is `models/checkpoints/<model>/ckpt.pt`. See `models/README.md` for details.

## Move ratings

Ratings append to `data/ratings.jsonl` by default. Set `RATINGS_PATH` to use another file.
