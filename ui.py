import gradio as gr
import time
import uuid
import random
import asyncio

from config import MAX_SEED, DEFAULT_GUIDANCE_SCALE, DEFAULT_STEPS, API_TIMEOUT, DEFAULT_NEGATIVE_PROMPT, MAX_IMAGE_SIZE, SUPPORTED_FORMATS

job_queue = None
results_store = None

def set_queues(q, r):
    global job_queue, results_store
    job_queue = q
    results_store = r

async def ui_infer(input_image, prompt, negative_prompt, seed, randomize_seed, guidance_scale, steps, output_extension, aspect_ratio, super_resolution, sr_scale, progress=gr.Progress(track_tqdm=True)):
    if input_image is None:
        raise gr.Error("Please upload an image.")
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    job_id = str(uuid.uuid4())
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    job_data = {
        "image": input_image, "prompt": prompt, "seed": seed,
        "guidance_scale": guidance_scale, "steps": steps, "negative_prompt": negative_prompt,
        "output_extension": output_extension, "aspect_ratio": aspect_ratio,
        "super_resolution": super_resolution, "sr_scale": sr_scale
    }
    job_queue.put((job_id, job_data))
    
    progress(0, desc="Request Queued...")
    start_time = time.time()
    while time.time() - start_time < API_TIMEOUT:
        if job_id in results_store:
            for i in progress.tqdm(range(100), desc="Generating Image..."):
                await asyncio.sleep(0.01) 
            result_payload = results_store.pop(job_id)

            if isinstance(result_payload, Exception):
                 raise gr.Error(f"Worker failed: {result_payload}")
            
            result_image, _ = result_payload

            if isinstance(result_image, Exception):
                raise gr.Error(f"Model inference failed: {result_image}")
            return result_image, seed
            
        queue_pos = job_queue.qsize() + 1
        progress(0.1, desc=f"Waiting in Queue (Position: {queue_pos})...")
        await asyncio.sleep(1)
        
    raise gr.Error("Request timed out. The server is busy. Please try again later.")

def create_ui():
    css="""#col-container { margin: 0 auto; max-width: 960px; }"""

    def update_output_size(image, aspect_ratio_choice, sr_scale_choice):
        if image is None:
            return "Upload an image first"
        
        width, height = image.size
        
        if aspect_ratio_choice == 'square':
            out_width, out_height = MAX_IMAGE_SIZE, MAX_IMAGE_SIZE
        else: # default
            if max(width, height) > MAX_IMAGE_SIZE:
                if width > height:
                    out_width = MAX_IMAGE_SIZE
                    out_height = int(height * MAX_IMAGE_SIZE / width)
                else:
                    out_height = MAX_IMAGE_SIZE
                    out_width = int(width * MAX_IMAGE_SIZE / height)
            else:
                out_width, out_height = width, height
                
        final_width = out_width * sr_scale_choice
        final_height = out_height * sr_scale_choice
        
        return f"{final_width} x {final_height} pixels"

    with gr.Blocks(css=css, title="Virtual Stager") as demo:
        with gr.Column(elem_id="col-container"):
            gr.Markdown("# Virtual Stager - FLUX.1 Kontext \nUpload an empty room image and provide a prompt to stage it.")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Empty Room Image", type="pil", height=400)
                    prompt = gr.Text(label="Staging Prompt", placeholder="e.g., A cozy Scandinavian living room with a fireplace")
                    run_button = gr.Button("Stage Room", variant="primary")
                    with gr.Accordion("Advanced Settings", open=False):
                        negative_prompt = gr.Text(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT, placeholder="e.g., ugly, deformed, blurry")
                        seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                        guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=DEFAULT_GUIDANCE_SCALE)
                        steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=DEFAULT_STEPS, step=1)
                        
                        gr.Markdown("---")
                        gr.Markdown("### Output Settings")
                        output_extension = gr.Radio(
                            list(SUPPORTED_FORMATS.keys()), 
                            label="Output File Type", 
                            value="jpeg"
                        )
                        aspect_ratio = gr.Radio(["default", "square"], label="Aspect Ratio", value="default", info="'default' preserves original aspect ratio. 'square' resizes to a square image.")
                        
                        gr.Markdown("### Super Resolution")
                        super_resolution = gr.Radio(["traditional"], label="Super Resolution Method", value="traditional", info="Deep learning based options will be added later.")
                        with gr.Row():
                            sr_scale = gr.Radio([2, 3], label="Super Resolution Scale (e.g., x2)", value=2)
                            output_size_info = gr.Textbox(label="Estimated Output Size", interactive=False, value="Upload an image first")

                with gr.Column():
                    result_image = gr.Image(label="Staged Result", interactive=False, height=400)
                    used_seed = gr.Number(label="Used Seed", interactive=False)
        
        run_button.click(
            fn=ui_infer,
            inputs=[input_image, prompt, negative_prompt, seed, randomize_seed, guidance_scale, steps, output_extension, aspect_ratio, super_resolution, sr_scale],
            outputs=[result_image, used_seed]
        )

        for component in [input_image, aspect_ratio, sr_scale]:
            component.change(
                fn=update_output_size,
                inputs=[input_image, aspect_ratio, sr_scale],
                outputs=[output_size_info]
            )

    return demo