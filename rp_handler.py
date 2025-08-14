import os
import base64
import io
import subprocess
import time
from PIL import Image
import runpod
from huggingface_hub import snapshot_download
from pathlib import Path
import torch

model = None

def initialize():
    global model
    start_time = time.time()
    source_engine_dir = Path("/runpod-volume/engines")
    local_engine_dir = Path("/app/engines")

    if source_engine_dir.exists():
        if not local_engine_dir.exists():
            local_engine_dir.mkdir(parents=True, exist_ok=True)
            rsync_command = [
                'rsync', '-ah', '--progress',
                str(source_engine_dir) + '/',
                str(local_engine_dir) + '/'
            ]
            result = subprocess.run(rsync_command, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"rsync failed with exit code {result.returncode}")
        os.environ["NETWORK_VOLUME_PATH"] = "/app"
    else:
        os.environ["NETWORK_VOLUME_PATH"] = "/runpod-volume"

    model_name = "black-forest-labs/FLUX.1-Kontext-dev"
    local_path = Path("./models/flux-dev-kontext")
    if not local_path.exists() or not any(local_path.iterdir()):
        snapshot_download(
            repo_id=model_name,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.safetensors", "*.onnx", "*.bin"]
        )

    from model_pipeline import StagingModel
    os.environ["HOME"] = "/app"
    Path.home = lambda: Path("/app")
    model = StagingModel()
    try:
        dummy_image = Image.new('RGB', (512, 512), 'white')
        _ = model.generate(
            prompt="warmup", input_image=dummy_image, seed=0,
            guidance_scale=1.0, steps=2, negative_prompt="",
            aspect_ratio="default", super_resolution="traditional",
            sr_scale=1, num_outputs=1
        )
        torch.cuda.synchronize()
    except Exception:
        pass
    end_time = time.time()
    return model

def handler(job: dict) -> dict:
    global model
    if model is None:
        model = job["model"]
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
    result_images, used_seeds = model.generate(**params)
    if isinstance(result_images, Exception):
        return {"error": f"Model generation failed: {str(result_images)}"}
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
    return {
        "images": base64_images,
        "seeds": used_seeds
    }

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler,
        "initializer": initialize
    })
