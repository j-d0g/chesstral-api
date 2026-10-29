# Model Management

This directory contains scripts for managing ChessGPT model checkpoints, including downloading from and uploading to Hugging Face Hub.

## Prerequisites

1. Install the API requirements:
```bash
pip install -r requirements.txt
```

2. Set up your Hugging Face token:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with write access
   - Save the token in a `.env` file in the project root:
     ```
     HF_TOKEN=your_token_here
     ```

## Usage

### Downloading Checkpoints

To download a specific checkpoint (aliases choose the highest iteration count):

```bash
.venv/bin/python models/download_checkpoint.py --model small-8
```

To download the latest checkpoint for each available model:

```bash
.venv/bin/python models/download_checkpoint.py
```

Arguments:
- `--model`: Specific model to download (optional, if not specified all models will be downloaded)
- `output_dir`: Optional destination (default: `models/checkpoints`)
- `--repo`: Hugging Face repository name (default: "jd0g/chess-gpt")

### Uploading Checkpoints

To upload a checkpoint to Hugging Face Hub:

```bash
python upload_checkpoint.py \
    --model_dir "checkpoints/small-24" \
    --repo_id "your-username/chess-gpt" \
    --token "your-huggingface-token" \
    --private
```

Arguments:
- `--model_dir`: Directory containing the checkpoint files (required)
- `--repo_id`: The repository ID on Hugging Face Hub (required)
- `--token`: Hugging Face API token (required)
- `--private`: Create a private repository (optional)

## Directory Structure

```
models/
├── README.md
├── download_checkpoint.py
└── upload_checkpoint.py
```

## Notes

- Make sure to keep your Hugging Face token secure and never commit it to version control
- The scripts will create the output directory if it doesn't exist
- For private repositories, you need to have the appropriate permissions on Hugging Face Hub
- Checkpoints are stored in `models/checkpoints/<model>/ckpt.pt`
- Exact checkpoint names such as `small-24-128K_iters` are accepted
- `meta.pkl` is read from `models/meta.pkl`