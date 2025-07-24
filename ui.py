import gradio as gr
import time
import uuid
import random

from config import MAX_SEED, DEFAULT_GUIDANCE_SCALE, DEFAULT_STEPS, API_TIMEOUT, DEFAULT_NEGATIVE_PROMPT

job_queue = None
results_store = None

def set_queues(q, r):
    global job_queue, results_store
    job_queue = q
    results_store = r

def ui_infer(input_image, prompt, negative_prompt, seed, randomize_seed, guidance_scale, steps, progress=gr.Progress(track_tqdm=True)):
    if input_image is None:
        raise gr.Error("Please upload an image.")
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    job_id = str(uuid.uuid4())
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    job_data = {
        "image": input_image, "prompt": prompt, "seed": seed,
        "guidance_scale": guidance_scale, "steps": steps, "negative_prompt": negative_prompt
    }
    job_queue.put((job_id, job_data))
    
    progress(0, desc="Request Queued...")
    start_time = time.time()
    while time.time() - start_time < API_TIMEOUT:
        if job_id in results_store:
            for i in progress.tqdm(range(100), desc="Generating Image..."):
                time.sleep(0.01)
            result = results_store.pop(job_id)
            if isinstance(result, Exception):
                raise gr.Error(f"Model inference failed: {result}")
            return result, seed
        queue_pos = job_queue.qsize() + 1
        progress(0.1, desc=f"Waiting in Queue (Position: {queue_pos})...")
        time.sleep(1)
        
    raise gr.Error("Request timed out. The server is busy. Please try again later.")

def create_ui():
    css="""#col-container { margin: 0 auto; max-width: 960px; }"""
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
                with gr.Column():
                    result_image = gr.Image(label="Staged Result", interactive=False, height=400)
                    used_seed = gr.Number(label="Used Seed", interactive=False)
        run_button.click(
            fn=ui_infer,
            inputs=[input_image, prompt, negative_prompt, seed, randomize_seed, guidance_scale, steps],
            outputs=[result_image, used_seed]
        )
    return demo