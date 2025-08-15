import os
import base64
import io
import random
from PIL import Image
import runpod

os.environ["HOME"] = "/app"
from pathlib import Path
Path.home = lambda: Path("/app")

from model_pipeline import StagingModel
from config import MAX_SEED, DEFAULT_GUIDANCE_SCALE, DEFAULT_STEPS, DEFAULT_NEGATIVE_PROMPT, SUPPORTED_FORMATS


print("Initializing StagingModel...")
model = StagingModel()
print("StagingModel initialized successfully.")



def handler(job):
    """
    The handler function called by RunPod for each job.
    It takes a job payload, runs inference, and returns the result.
    """
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

    params = {
        "prompt": prompt,
        "input_image": input_image,
        "negative_prompt": job_input.get('negative_prompt', DEFAULT_NEGATIVE_PROMPT),
        "seed": int(job_input.get('seed', -1)),
        "guidance_scale": float(job_input.get('guidance_scale', DEFAULT_GUIDANCE_SCALE)),
        "steps": int(job_input.get('steps', DEFAULT_STEPS)),
        "num_outputs": int(job_input.get('num_outputs', 1)),
        "aspect_ratio": job_input.get('aspect_ratio', "default"),
        "super_resolution": job_input.get('super_resolution', "traditional"),
        "sr_scale": int(job_input.get('sr_scale', 2))
    }
    
    if params["seed"] == -1:
        params["seed"] = random.randint(0, MAX_SEED)

    print(f"Starting generation with seed: {params['seed']}")
    results = model.generate(**params)

    if isinstance(results, Exception):
        print(f"Model generation failed: {results}")
        return {"error": f"Model generation failed: {results}"}

    output_extension = job_input.get('output_extension', 'jpeg').lower()
    format_info = SUPPORTED_FORMATS.get(output_extension, SUPPORTED_FORMATS['jpeg'])
    image_format = format_info['format']
    
    base64_images = []
    for image in results:
        if image_format in ['JPEG', 'BMP'] and image.mode == 'RGBA':
            image = image.convert('RGB')
            
        buffered = io.BytesIO()
        image.save(buffered, format=image_format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_images.append(img_str)

    print(f"Generation complete. Returning {len(base64_images)} image(s).")
    
    return {
        "images_base64": base64_images,
        "seed": params["seed"]
    }


if __name__ == "__main__":
    print("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})