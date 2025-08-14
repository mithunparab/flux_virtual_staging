import os
import base64
import io
import time
from PIL import Image
import runpod
from huggingface_hub import snapshot_download
from pathlib import Path
import torch
import traceback

model = None

def _initialize():
    global model
    start_time = time.time()
    try:
        os.environ["NETWORK_VOLUME_PATH"] = "/app"
        local_model_dir = Path("./models/flux-dev-kontext")
        if not local_model_dir.exists() or not any(local_model_dir.iterdir()):
            model_name = "black-forest-labs/FLUX.1-Kontext-dev"
            snapshot_download(
                repo_id=model_name,
                local_dir=local_model_dir,
                local_dir_use_symlinks=False,
                ignore_patterns=["*.safetensors", "*.onnx", "*.bin"]
            )
        from model_pipeline import StagingModel
        os.environ["HOME"] = "/app"
        Path.home = lambda: Path("/app")
        model = StagingModel()
        dummy_image = Image.new('RGB', (512, 512), 'white')
        _ = model.generate(
            prompt="warmup", input_image=dummy_image, seed=0,
            guidance_scale=1.0, steps=2, negative_prompt="",
            aspect_ratio="default", super_resolution="traditional",
            sr_scale=1, num_outputs=1
        )
        torch.cuda.synchronize()
        end_time = time.time()
    except Exception:
        traceback.print_exc()
        model = None

def handler(job: dict) -> dict:
    global model
    if model is None:
        _initialize()
        if model is None:
            return {"error": "Model failed to initialize. Please check the pod logs for errors."}
    job_input: dict = job.get('input', {})
    image_base64: str = job.get('input', {}).get('image_base64')
    if not image_base64: return {"error": "Missing 'image_base64' in input."}
    prompt: str = job.get('input', {}).get('prompt')
    if not prompt: return {"error": "Missing 'prompt' in input."}
    try:
        image_bytes: bytes = base64.b64decode(image_base64)
        input_image: Image.Image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return {"error": f"Invalid base64 string or image format: {e}"}
    from config import DEFAULT_GUIDANCE_SCALE, DEFAULT_STEPS, DEFAULT_NEGATIVE_PROMPT, SUPPORTED_FORMATS
    params: dict = {
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
    if isinstance(result_images, Exception): return {"error": f"Model generation failed: {str(result_images)}"}
    output_extension: str = job_input.get('output_extension', 'jpeg').lower()
    format_info: dict = SUPPORTED_FORMATS.get(output_extension, SUPPORTED_FORMATS['jpeg'])
    image_format: str = format_info['format']
    base64_images: list[str] = []
    for img in result_images:
        if image_format in ['JPEG', 'BMP'] and img.mode == 'RGBA': img = img.convert('RGB')
        buffered: io.BytesIO = io.BytesIO()
        img.save(buffered, format=image_format)
        img_str: str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_images.append(img_str)
    return {"images": base64_images, "seeds": used_seeds}

if __name__ == "__main__":
    print("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
