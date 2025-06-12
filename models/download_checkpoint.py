#!/usr/bin/env python3
"""
Download ChessGPT checkpoints from Hugging Face Hub.
Usage: python download_checkpoint.py [output_dir] [repo_name]
Example: python download_checkpoint.py ./checkpoints jd0g/chess-gpt
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

def download_checkpoint(model_name, output_dir, repo_name="jd0g/chess-gpt"):
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
    
    # Path in repo for the checkpoint
    path_in_repo = f"checkpoints/{model_name}/ckpt.pt"
    output_path = os.path.join(output_dir, model_name, "ckpt.pt")
    
    print(f"Downloading checkpoint {model_name} from {repo_name}...")
    
    try:
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download the file from the Hugging Face Hub
        output_file = hf_hub_download(
            repo_id=repo_name,
            filename=path_in_repo,
            token=token,
            local_dir=os.path.dirname(output_path),
            local_dir_use_symlinks=False,
            force_download=True
        )
        
        print(f"Checkpoint {model_name} downloaded successfully to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error downloading checkpoint {model_name}: {e}")
        return False

def download_all_checkpoints(output_dir, repo_name="jd0g/chess-gpt"):
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
        model_names = [f.split("/")[1] for f in checkpoint_files]
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
    parser.add_argument("output_dir", nargs="?", default="./checkpoints", 
                        help="Directory where to save the downloaded checkpoints")
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