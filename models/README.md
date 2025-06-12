# Model Management

This directory contains scripts for managing ChessGPT model checkpoints, including downloading from and uploading to Hugging Face Hub.

## Prerequisites

1. Install required packages:
```bash
pip install huggingface_hub transformers python-dotenv
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

To download a specific checkpoint:

```bash
python download_checkpoint.py \
    --model "small-24" \
    --output_dir "checkpoints" \
    --repo "jd0g/chess-gpt"
```

To download all available checkpoints:

```bash
python download_checkpoint.py \
    --output_dir "checkpoints" \
    --repo "jd0g/chess-gpt"
```

Arguments:
- `--model`: Specific model to download (optional, if not specified all models will be downloaded)
- `--output_dir`: Directory to save the checkpoints (default: "checkpoints")
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
- Checkpoints are stored in the `checkpoints/` directory at the project root 