import gradio as gr
import time
import uuid
import random
import asyncio
import io
import base64
import asyncio
from PIL import Image
from config import MAX_SEED, DEFAULT_GUIDANCE_SCALE, DEFAULT_STEPS, API_TIMEOUT, DEFAULT_NEGATIVE_PROMPT, MAX_IMAGE_SIZE, SUPPORTED_FORMATS

job_queue = None
results_store = None

def set_queues(q, r):
    global job_queue, results_store
    job_queue = q
    results_store = r

async def ui_infer(input_image, prompt, negative_prompt, seed, randomize_seed, guidance_scale, steps, num_outputs, output_extension, aspect_ratio, super_resolution, sr_scale, progress=gr.Progress(track_tqdm=True)):
    if input_image is None:
        raise gr.Error("Please upload an image.")
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    job_id = str(uuid.uuid4())
    final_seed = -1 
    if not randomize_seed:
        final_seed = seed

    job_data = {
        "image": input_image, "prompt": prompt, "seed": final_seed,
        "guidance_scale": guidance_scale, "steps": steps, "negative_prompt": negative_prompt,
        "num_outputs": num_outputs,
        "output_extension": output_extension, "aspect_ratio": aspect_ratio,
        "super_resolution": super_resolution, "sr_scale": sr_scale
    }
    job_queue.put((job_id, job_data))
    
    progress(0, desc="Request Queued...")
    start_time = time.time()
    while time.time() - start_time < API_TIMEOUT:
        if job_id in results_store:
            for i in progress.tqdm(range(100), desc=f"Generating Batch of {num_outputs}..."):
                await asyncio.sleep(0.01)
                
            result_payload = results_store.pop(job_id)
            
            if isinstance(result_payload, Exception):
                 raise gr.Error(f"Worker failed: {result_payload}")
            
            base64_images_str = result_payload.get("images", [])
            used_seeds = result_payload.get("seeds", [])

            final_images = []
            for b64_str in base64_images_str:
                image_bytes = base64.b64decode(b64_str)
                image = Image.open(io.BytesIO(image_bytes))
                final_images.append(image)

            return final_images, used_seeds

            
        queue_pos = job_queue.qsize() + 1
        progress(0.1, desc=f"Waiting in Queue (Position: {queue_pos})...")
        await asyncio.sleep(1)
        
    raise gr.Error("Request timed out. The server is busy. Please try again later.")

def create_ui():
    css="""#col-container { margin: 0 auto; max-width: 960px; } #gallery { min-height: 512px; }"""

    with gr.Blocks(css=css, title="Virtual Stager") as demo:
        with gr.Column(elem_id="col-container"):
            gr.Markdown("# Virtual Stager - FLUX.1 Kontext \nUpload an empty room image and provide a prompt to stage it.")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Empty Room Image", type="pil", height=400)
                    prompt = gr.Text(label="Staging Prompt", placeholder="e.g., A cozy Scandinavian living room with a fireplace")
                    run_button = gr.Button("Stage Room", variant="primary")
                    with gr.Accordion("Advanced Settings", open=False):

                        num_outputs = gr.Slider(label="Number of Images", minimum=1, maximum=8, step=1, value=1)
                        negative_prompt = gr.Text(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT, placeholder="e.g., ugly, deformed, blurry")
                        seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                        guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=DEFAULT_GUIDANCE_SCALE)
                        steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=DEFAULT_STEPS, step=1)
                        gr.Markdown("---")
                        gr.Markdown("### Output Settings")
                        output_extension = gr.Radio(list(SUPPORTED_FORMATS.keys()), label="Output File Type", value="jpeg")
                        aspect_ratio = gr.Radio(["default", "square"], label="Aspect Ratio", value="default")
                        super_resolution = gr.Radio(["traditional"], label="Super Resolution Method", value="traditional")
                        sr_scale = gr.Radio([2, 3], label="Super Resolution Scale (e.g., x2)", value=2)

                with gr.Column():
                    result_gallery = gr.Gallery(label="Staged Results", elem_id="gallery", columns=2, object_fit="contain", height=512)
                    used_seeds_output = gr.JSON(label="Used Seeds")
        
        run_button.click(
            fn=ui_infer,
            inputs=[input_image, prompt, negative_prompt, seed, randomize_seed, guidance_scale, steps, num_outputs, output_extension, aspect_ratio, super_resolution, sr_scale],
            outputs=[result_gallery, used_seeds_output]
        )

    return demo