"""
Utility functions for handling model training metadata and dataset operations.
"""

import os
import json
import hashlib
import pandas as pd
from rich.console import Console

# Initialize console for colored output
console = Console()


def calculate_dataset_hash(dataset_path: str) -> str:
    """Calculate a hash of the dataset file to uniquely identify it"""
    try:
        hash_md5 = hashlib.md5()
        with open(dataset_path, "rb") as f:
            # Read file in chunks to handle large datasets
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        console.print(f"[bold yellow][13][/bold yellow] Could not calculate dataset hash: {e}")
        return None


def get_training_metadata_path(models_dir: str, model_id: str) -> str:
    """Get path to training metadata file"""
    safe_model_id = model_id.replace("/", "_").replace("\\", "_")
    metadata_file = f"{safe_model_id}_training_metadata.json"
    return os.path.join(models_dir, metadata_file)


def load_training_metadata(models_dir: str, model_id: str) -> dict:
    """Load training metadata if it exists"""
    metadata_path = get_training_metadata_path(models_dir, model_id)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[info] Could not load training metadata: {e}")
    return {}


def save_training_metadata(models_dir: str, model_id: str, dataset_hash: str, merged_model_dir: str):
    """Save training metadata for future reference"""
    metadata_path = get_training_metadata_path(models_dir, model_id)
    metadata = load_training_metadata(models_dir, model_id)

    metadata[dataset_hash] = {
        "dataset_hash": dataset_hash,
        "merged_model_dir": merged_model_dir,
        "training_date": str(pd.Timestamp.now()),
        "model_id": model_id
    }

    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        console.print(f"[bold green][1][/bold green] Training metadata saved")
    except Exception as e:
        console.print(f"[bold yellow][13][/bold yellow] Could not save training metadata: {e}")


def check_existing_training(models_dir: str, model_id: str, dataset_hash: str) -> str:
    """Check if model has already been trained for this dataset"""
    if not dataset_hash:
        return None

    metadata = load_training_metadata(models_dir, model_id)

    if dataset_hash in metadata:
        merged_model_dir = metadata[dataset_hash]["merged_model_dir"]
        if os.path.exists(merged_model_dir):
            training_date = metadata[dataset_hash].get("training_date", "unknown")
            console.print(f"[bold green][1][/bold green] Found existing training for this dataset")
            console.print(f"[bold cyan][14][/bold cyan] Training date: {training_date}")
            console.print(f"[bold cyan][14][/bold cyan] Model location: {merged_model_dir}")
            return merged_model_dir
        else:
            console.print(f"[bold yellow][13][/bold yellow] Training metadata found but model directory missing: {merged_model_dir}")
            # Clean up invalid metadata entry
            try:
                del metadata[dataset_hash]
                metadata_path = get_training_metadata_path(models_dir, model_id)
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            except:
                pass

    return None
