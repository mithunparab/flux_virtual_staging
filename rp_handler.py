import os
import base64
import io
from PIL import Image
import runpod
from huggingface_hub import snapshot_download
from pathlib import Path

model = None

# rp_handler.py

import os
import base64
import io
import shutil  # <-- Import shutil
from PIL import Image
import runpod
from huggingface_hub import snapshot_download
from pathlib import Path

model = None

def initialize_model():
    global model
    print("Cold start: Initializing StagingModel...")

    source_engine_dir = Path("/runpod-volume/engines")
    local_engine_dir = Path("/app/engines")

    if source_engine_dir.exists():
        if not local_engine_dir.exists():
            print(f"Copying engines from {source_engine_dir} to local storage {local_engine_dir} for performance...")
            shutil.copytree(source_engine_dir, local_engine_dir)
            print("Engine copy complete.")
        else:
            print("Local engine directory already exists. Skipping copy.")
        os.environ["NETWORK_VOLUME_PATH"] = "/app"
    else:
        print(f"WARNING: Source engine directory {source_engine_dir} not found.")
    
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise ValueError("FATAL: HUGGING_FACE_HUB_TOKEN secret not found in runtime environment.")

    model_name = "black-forest-labs/FLUX.1-Kontext-dev"
    local_path = Path("./models/flux-dev-kontext")
    
    print(f"Downloading base model '{model_name}' to {local_path}...")
    snapshot_download(
        repo_id=model_name,
        local_dir=local_path,
        token=hf_token,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.safetensors", "*.onnx", "*.bin"]
    )
    print("Base model download complete.")

    from model_pipeline import StagingModel
    os.environ["HOME"] = "/app"
    Path.home = lambda: Path("/app")
    
    model = StagingModel()
    print("StagingModel initialized successfully.")


def handler(job):
    global model

    if model is None:
        initialize_model()

    job_input = job.get('input', {})

    image_base64 = job_input.get('image_base64')
    if not image_base64:
        return {"error": "Missing 'image_base64' in input."}

    prompt = job_input.get('prompt')
    if not prompt:
        return {"error": "Missing 'prompt' in input."}

    try:
        image_bytes = base64.b64decode(image_base64)
        input_image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return {"error": f"Invalid base64 string or image format: {e}"}

    from config import DEFAULT_GUIDANCE_SCALE, DEFAULT_STEPS, DEFAULT_NEGATIVE_PROMPT, SUPPORTED_FORMATS

    params = {
        "prompt": prompt,
        "input_image": input_image,
        "negative_prompt": job_input.get('negative_prompt', DEFAULT_NEGATIVE_PROMPT),
        "seed": int(job_input.get('seed', -1)),
        "guidance_scale": float(job_input.get('guidance_scale', DEFAULT_GUIDANCE_SCALE)),
        "steps": int(job_input.get('steps', DEFAULT_STEPS)),
        "aspect_ratio": job_input.get('aspect_ratio', "default"),
        "super_resolution": job_input.get('super_resolution', "traditional"),
        "sr_scale": int(job_input.get('sr_scale', 2)),
        "num_outputs": int(job_input.get('num_outputs', 1))
    }

    print(f"Starting batch generation of {params['num_outputs']} with base seed: {params['seed']}")
    result_images, used_seeds = model.generate(**params)

    if isinstance(result_images, Exception):
        print(f"Model generation failed: {result_images}")
        return {"error": f"Model generation failed: {result_images}"}

    output_extension = job_input.get('output_extension', 'jpeg').lower()
    format_info = SUPPORTED_FORMATS.get(output_extension, SUPPORTED_FORMATS['jpeg'])
    image_format = format_info['format']
    
    base64_images = []
    for img in result_images:
        if image_format in ['JPEG', 'BMP'] and img.mode == 'RGBA':
            img = img.convert('RGB')
        
        buffered = io.BytesIO()
        img.save(buffered, format=image_format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_images.append(img_str)

    print(f"Batch generation of {len(base64_images)} images complete.")
    
    return {
        "images": base64_images,
        "seeds": used_seeds
    }

if __name__ == "__main__":
    print("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})