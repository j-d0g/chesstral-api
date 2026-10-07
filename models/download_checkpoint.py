#!/usr/bin/env python3
"""
Download ChessGPT checkpoints from Hugging Face Hub.
Usage: python download_checkpoint.py [output_dir] [repo_name]
Example: python download_checkpoint.py ./checkpoints jd0g/chess-gpt
"""

import os
import sys
import argparse
import re
import shutil
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download, list_repo_files
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models" / "checkpoints"
load_dotenv(REPO_ROOT / ".env")


def resolve_checkpoint_name(model_name: str, files: list[str]) -> Optional[str]:
    exact_path = f"checkpoints/{model_name}/ckpt.pt"
    if exact_path in files:
        return model_name

    alias_pattern = re.compile(
        rf"^checkpoints/{re.escape(model_name)}-(\d+)[kK]_iters/ckpt\.pt$"
    )
    candidates = []
    for path in files:
        match = alias_pattern.match(path)
        if match:
            candidates.append((int(match.group(1)) * 1000, path.split("/")[1]))

    if not candidates:
        return None
    return max(candidates)[1]


def download_checkpoint(model_name, output_dir=DEFAULT_OUTPUT_DIR, repo_name="jd0g/chess-gpt"):
    """
    Download a checkpoint from Hugging Face Hub
    Args:
        model_name: Name of the model to download (e.g., small-24, large-24-184K_iters)
        output_dir: Directory where to save the downloaded checkpoint
        repo_name: Name of the Hugging Face repository
    """
    # Get token from environment variable
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not found. Please set it in your .env file.")
        return False
    
    print(f"Downloading checkpoint {model_name} from {repo_name}...")
    
    try:
        files = list_repo_files(repo_id=repo_name, token=token)
        resolved_name = resolve_checkpoint_name(model_name, files)
        if resolved_name is None:
            print(f"No checkpoint found for model {model_name} in {repo_name}")
            return False

        output_path = Path(output_dir) / model_name / "ckpt.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        downloaded_file = hf_hub_download(
            repo_id=repo_name,
            filename=f"checkpoints/{resolved_name}/ckpt.pt",
            token=token,
        )
        shutil.copy2(downloaded_file, output_path)
        
        print(f"Checkpoint {model_name} downloaded successfully to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error downloading checkpoint {model_name}: {e}")
        return False

def download_all_checkpoints(
    output_dir=DEFAULT_OUTPUT_DIR,
    repo_name="jd0g/chess-gpt",
):
    """
    Download all available checkpoints from Hugging Face Hub
    Args:
        output_dir: Directory where to save the downloaded checkpoints
        repo_name: Name of the Hugging Face repository
    """
    # Get token from environment variable
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not found. Please set it in your .env file.")
        return False
    
    print(f"Listing available checkpoints in {repo_name}...")
    
    try:
        # List all files in the repository
        files = list_repo_files(repo_id=repo_name, token=token)
        
        # Filter for checkpoint files
        checkpoint_files = [f for f in files if f.startswith("checkpoints/") and f.endswith("/ckpt.pt")]
        
        if not checkpoint_files:
            print(f"No checkpoints found in repository {repo_name}")
            return False
        
        # Extract model names from paths
        model_names = sorted({f.split("/")[1] for f in checkpoint_files})
        print(f"Found {len(model_names)} checkpoints: {', '.join(model_names)}")
        
        # Download each checkpoint
        success = True
        for model_name in model_names:
            if not download_checkpoint(model_name, output_dir, repo_name):
                success = False
        
        return success
        
    except Exception as e:
        print(f"Error listing checkpoints: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ChessGPT checkpoints from Hugging Face Hub")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where to save the downloaded checkpoints",
    )
    parser.add_argument("--repo", default="jd0g/chess-gpt", help="Hugging Face repository name")
    parser.add_argument("--model", help="Specific model to download (if not specified, all models will be downloaded)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download specific model or all models
    if args.model:
        success = download_checkpoint(args.model, args.output_dir, args.repo)
    else:
        success = download_all_checkpoints(args.output_dir, args.repo)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)