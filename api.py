import uuid
import time
import io
import random
import base64
import asyncio
from PIL import Image
from enum import Enum 

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import MAX_SEED, DEFAULT_GUIDANCE_SCALE, DEFAULT_STEPS, API_TIMEOUT, DEFAULT_NEGATIVE_PROMPT, MAX_IMAGE_SIZE, SUPPORTED_FORMATS

job_queue = None
results_store = None
router = APIRouter()

OutputExtensionEnum = Enum("OutputExtensionEnum", {k: k for k in SUPPORTED_FORMATS.keys()})

class Base64StageRequest(BaseModel):
    image_base64: str = Field(..., description="A base64 encoded string of the input image.")
    prompt: str = Field(..., description="A description of the desired staging style.")
    negative_prompt: str = Field(default=DEFAULT_NEGATIVE_PROMPT, description="Items to avoid in the generated image.")
    seed: int | None = Field(default=None, description=f"Seed for reproducibility. If null or -1, a random seed is used. Max: {MAX_SEED}")
    guidance_scale: float = Field(default=DEFAULT_GUIDANCE_SCALE, description="Controls how much the prompt guides the image generation.")
    steps: int = Field(default=DEFAULT_STEPS, description="Number of inference steps.")
    output_extension: OutputExtensionEnum = Field(default='jpeg', description="Output image format.")
    aspect_ratio: str = Field(default="default", description="Aspect ratio for the output image: 'default' or 'square'.")
    super_resolution: str = Field(default="traditional", description="Super resolution method. Currently only 'traditional' is supported.")
    sr_scale: int = Field(default=2, description="Super resolution scale factor: 2 or 3.")

def set_queues(q, r):
    global job_queue, results_store
    job_queue = q
    results_store = r

async def _process_job_and_get_result(job_data: dict):
    job_id = str(uuid.uuid4())
    job_queue.put((job_id, job_data))
    start_time = time.time()
    while time.time() - start_time < API_TIMEOUT:
        if job_id in results_store:
            result_data = results_store.pop(job_id)

            if isinstance(result_data, Exception):
                raise HTTPException(status_code=500, detail=f"Model inference failed: {result_data}")

            result_image, output_extension_value = result_data
            
            if isinstance(result_image, Image.Image):
                requested_format = output_extension_value.lower()
                format_info = SUPPORTED_FORMATS.get(requested_format, SUPPORTED_FORMATS['jpeg']) # Use dict from config

                image_format = format_info['format']
                media_type = format_info['media_type']
                
                if image_format in ['JPEG', 'BMP']:
                    if result_image.mode == 'RGBA':
                        result_image = result_image.convert('RGB')

                img_byte_arr = io.BytesIO()
                result_image.save(img_byte_arr, format=image_format)
                img_byte_arr.seek(0)
                
                return StreamingResponse(img_byte_arr, media_type=media_type)
            else:
                raise HTTPException(status_code=500, detail=f"Model inference failed: {result_image}")

        await asyncio.sleep(0.1)
    raise HTTPException(status_code=504, detail="Request timed out. The server is busy. Please try again later.")

@router.post("/stage_upload", summary="Stage an Image via File Upload")
async def stage_image_upload(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(default=DEFAULT_NEGATIVE_PROMPT),
    seed: int = Form(None),
    guidance_scale: float = Form(default=DEFAULT_GUIDANCE_SCALE),
    steps: int = Form(default=DEFAULT_STEPS),
    output_extension: OutputExtensionEnum = Form(default='jpeg'),
    aspect_ratio: str = Form(default="default"),
    super_resolution: str = Form(default="traditional"),
    sr_scale: int = Form(default=2)
):
    try:
        input_image = Image.open(image.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file provided.")
    if seed is None or seed == -1:
        seed = random.randint(0, MAX_SEED)
    job_data = {
        "image": input_image, "prompt": prompt, "seed": seed,
        "guidance_scale": guidance_scale, "steps": steps, "negative_prompt": negative_prompt,
        "output_extension": output_extension.value,
        "aspect_ratio": aspect_ratio,
        "super_resolution": super_resolution, "sr_scale": sr_scale
    }
    return await _process_job_and_get_result(job_data)

@router.post("/stage_base64", summary="Stage an Image via Base64 JSON")
async def stage_image_base64(request: Base64StageRequest):
    try:
        image_bytes = base64.b64decode(request.image_base64)
        input_image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 string or image format.")
    seed = request.seed
    if seed is None or seed == -1:
        seed = random.randint(0, MAX_SEED)
    job_data = {
        "image": input_image, "prompt": request.prompt, "seed": seed,
        "guidance_scale": request.guidance_scale, "steps": request.steps, "negative_prompt": request.negative_prompt,
        "output_extension": request.output_extension.value,
        "aspect_ratio": request.aspect_ratio,
        "super_resolution": request.super_resolution, "sr_scale": request.sr_scale
    }
    return await _process_job_and_get_result(job_data)