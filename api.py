import uuid
import time
import io
import random
import base64
import asyncio
from PIL import Image
from enum import Enum 
from typing import List

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse 
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
    num_outputs: int = Field(default=1, description="Number of images to generate in a batch.") # New
    output_extension: OutputExtensionEnum = Field(default='jpeg', description="Output image format.")
    aspect_ratio: str = Field(default="default", description="Aspect ratio for the output image: 'default' or 'square'.")
    super_resolution: str = Field(default="traditional", description="Super resolution method.")
    sr_scale: int = Field(default=2, description="Super resolution scale factor: 2 or 3.")

class BatchStageResponse(BaseModel):
    images: List[str] = Field(..., description="List of Base64 encoded result images.")
    seeds: List[int] = Field(..., description="List of seeds used for each corresponding image.")

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
            result_payload, output_extension_value = results_store.pop(job_id)

            if isinstance(result_payload, Exception):
                raise HTTPException(status_code=500, detail=f"Model inference failed: {result_payload}")

            result_images, used_seeds = result_payload
            
            base64_images = []
            requested_format = output_extension_value.lower()
            image_format = SUPPORTED_FORMATS.get(requested_format, SUPPORTED_FORMATS['jpeg'])['format']

            for img in result_images:
                if isinstance(img, Image.Image):
                    if image_format in ['JPEG', 'BMP'] and img.mode == 'RGBA':
                        img = img.convert('RGB')
                    
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format=image_format)
                    img_byte_arr.seek(0)
                    base64_images.append(base64.b64encode(img_byte_arr.getvalue()).decode("utf-8"))
                else:
                    raise HTTPException(status_code=500, detail="Model returned an invalid object in the image list.")
            
            return JSONResponse(content={"images": base64_images, "seeds": used_seeds})

        await asyncio.sleep(0.1)
    raise HTTPException(status_code=504, detail="Request timed out. The server is busy. Please try again later.")


@router.post("/stage_upload", summary="Stage an Image via File Upload", response_model=BatchStageResponse)
async def stage_image_upload(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(default=DEFAULT_NEGATIVE_PROMPT),
    seed: int = Form(None),
    guidance_scale: float = Form(default=DEFAULT_GUIDANCE_SCALE),
    steps: int = Form(default=DEFAULT_STEPS),
    num_outputs: int = Form(default=1),
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
        seed = -1
    
    job_data = {
        "image": input_image, "prompt": prompt, "seed": seed,
        "guidance_scale": guidance_scale, "steps": steps, "negative_prompt": negative_prompt,
        "num_outputs": num_outputs,
        "output_extension": output_extension.value,
        "aspect_ratio": aspect_ratio,
        "super_resolution": super_resolution, "sr_scale": sr_scale
    }
    return await _process_job_and_get_result(job_data)

@router.post("/stage_base64", summary="Stage an Image via Base64 JSON", response_model=BatchStageResponse)
async def stage_image_base64(request: Base64StageRequest):
    try:
        image_bytes = base64.b64decode(request.image_base64)
        input_image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 string or image format.")
    
    seed = request.seed
    if seed is None or seed == -1:
        seed = -1 
        
    job_data = {
        "image": input_image, "prompt": request.prompt, "seed": seed,
        "guidance_scale": request.guidance_scale, "steps": request.steps, "negative_prompt": request.negative_prompt,
        "num_outputs": request.num_outputs, # New
        "output_extension": request.output_extension.value,
        "aspect_ratio": request.aspect_ratio,
        "super_resolution": request.super_resolution, "sr_scale": request.sr_scale
    }
    return await _process_job_and_get_result(job_data)