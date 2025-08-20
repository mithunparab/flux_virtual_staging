import os
from pathlib import Path
import runpod

model = None

def _initialize():
    """Performs the Python-level model loading and warmup."""
    global model
    import time
    from PIL import Image
    import torch
    import traceback

    print("--- [HANDLER COLD START] Performing one-time model initialization... ---")
    start_time = time.time()
    try:
        from model_pipeline import StagingModel
        os.environ["HOME"] = "/app"
        Path.home = lambda: Path("/app")
        
        model = StagingModel()
        print("[INIT] StagingModel object created successfully.")

        print("[INIT] Running warm-up inference...")
        dummy_image = Image.new('RGB', (512, 512), 'white')
        _ = model.generate(
            prompt="warmup", input_image=dummy_image, seed=0,
            guidance_scale=1.0, steps=2, negative_prompt="",
            aspect_ratio="default", super_resolution="traditional", sr_scale=1, num_outputs=1
        )
        torch.cuda.synchronize()
        print("[INIT] Warm-up finished.")
        end_time = time.time()
        print(f"--- [INITIALIZATION SUCCESS] Cold start finished in {end_time - start_time:.2f}s ---")

    except Exception:
        print("--- [INITIALIZATION FAILED] An error occurred during model loading. ---")
        traceback.print_exc()
        model = None

def handler(job: dict) -> dict:
    """The main handler function. Initializes the model on the first run."""
    global model
    import base64, io
    from config import DEFAULT_GUIDANCE_SCALE, DEFAULT_STEPS, DEFAULT_NEGATIVE_PROMPT, SUPPORTED_FORMATS
    from PIL import Image

    if model is None:
        _initialize()
        if model is None:
            return {"error": "Model failed to initialize. Check pod logs for errors."}
    
    job_input = job.get('input', {})
    image_base64 = job_input.get('image_base64')
    if not image_base64: return {"error": "Missing 'image_base64' in input."}
    prompt = job_input.get('prompt')
    if not prompt: return {"error": "Missing 'prompt' in input."}
    try:
        image_bytes = base64.b64decode(image_base64)
        input_image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return {"error": f"Invalid base64 string or image format: {e}"}
    
    params = { "prompt": prompt, "input_image": input_image, "negative_prompt": job_input.get('negative_prompt', DEFAULT_NEGATIVE_PROMPT), "seed": int(job_input.get('seed', -1)), "guidance_scale": float(job_input.get('guidance_scale', DEFAULT_GUIDANCE_SCALE)), "steps": int(job_input.get('steps', DEFAULT_STEPS)), "aspect_ratio": job_input.get('aspect_ratio', "default"), "super_resolution": job_input.get('super_resolution', "traditional"), "sr_scale": int(job_input.get('sr_scale', 2)), "num_outputs": int(job_input.get('num_outputs', 1)) }
    
    result_images, used_seeds = model.generate(**params)
    if isinstance(result_images, Exception): return {"error": f"Model generation failed: {str(result_images)}"}
    
    output_extension = job_input.get('output_extension', 'jpeg').lower()
    format_info = SUPPORTED_FORMATS.get(output_extension, SUPPORTED_FORMATS['jpeg'])
    image_format = format_info['format']
    base64_images = []
    for img in result_images:
        if image_format in ['JPEG', 'BMP'] and img.mode == 'RGBA': img = img.convert('RGB')
        buffered = io.BytesIO()
        img.save(buffered, format=image_format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_images.append(img_str)
        
    return {"images": base64_images, "seeds": used_seeds}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})