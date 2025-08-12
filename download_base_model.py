import os
from huggingface_hub import snapshot_download
from pathlib import Path

if __name__ == "__main__":

    model_name = "black-forest-labs/FLUX.1-Kontext-dev"
    local_path = Path("./models/flux-dev-kontext")

    print(f"Downloading base model '{model_name}' for runtime...")
    snapshot_download(
        repo_id=model_name,
        local_dir=local_path,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.safetensors", "*.onnx", "*.bin"]
    )
    print(f"Base model files prepared in {local_path}")