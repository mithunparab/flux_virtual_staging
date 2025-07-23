import uuid
import time
import io
import random
import base64
import asyncio
from PIL import Image

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import MAX_SEED, DEFAULT_GUIDANCE_SCALE, DEFAULT_STEPS, API_TIMEOUT, DEFAULT_NEGATIVE_PROMPT

job_queue = None
results_store = None
router = APIRouter()

class Base64StageRequest(BaseModel):
    image_base64: str = Field(..., description="A base64 encoded string of the input image.")
    prompt: str = Field(..., description="A description of the desired staging style.")
    negative_prompt: str = Field(default=DEFAULT_NEGATIVE_PROMPT, description="Items to avoid in the generated image.")
    seed: int | None = Field(default=None, description=f"Seed for reproducibility. If null or -1, a random seed is used. Max: {MAX_SEED}")
    guidance_scale: float = Field(default=DEFAULT_GUIDANCE_SCALE, description="Controls how much the prompt guides the image generation.")
    steps: int = Field(default=DEFAULT_STEPS, description="Number of inference steps.")

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
            result = results_store.pop(job_id)
            if isinstance(result, Image.Image):
                img_byte_arr = io.BytesIO()
                result.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                return StreamingResponse(img_byte_arr, media_type="image/png")
            else:
                raise HTTPException(status_code=500, detail=f"Model inference failed: {result}")
        await asyncio.sleep(0.1)
    raise HTTPException(status_code=504, detail="Request timed out. The server is busy. Please try again later.")

@router.post("/stage_upload", summary="Stage an Image via File Upload")
async def stage_image_upload(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(default=DEFAULT_NEGATIVE_PROMPT),
    seed: int = Form(None),
    guidance_scale: float = Form(default=DEFAULT_GUIDANCE_SCALE),
    steps: int = Form(default=DEFAULT_STEPS)
):
    try:
        input_image = Image.open(image.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file provided.")
    if seed is None or seed == -1:
        seed = random.randint(0, MAX_SEED)
    job_data = {
        "image": input_image, "prompt": prompt, "seed": seed,
        "guidance_scale": guidance_scale, "steps": steps, "negative_prompt": negative_prompt
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
        "guidance_scale": request.guidance_scale, "steps": request.steps, "negative_prompt": request.negative_prompt
    }
    return await _process_job_and_get_result(job_data)