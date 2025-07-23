import time
from queue import Queue
from model_pipeline import StagingModel

def processing_worker(model: StagingModel, job_queue: Queue, results_store: dict):
    print("Worker thread started. Waiting for jobs...")
    while True:
        try:
            job_id, job_data = job_queue.get()
            print(f"Worker received job: {job_id}")

            result = model.generate(
                prompt=job_data["prompt"],
                input_image=job_data["image"],
                seed=job_data["seed"],
                guidance_scale=job_data["guidance_scale"],
                steps=job_data["steps"],
                negative_prompt=job_data["negative_prompt"]
            )
            
            results_store[job_id] = result
            print(f"Worker finished job: {job_id}")

        except Exception as e:
            print(f"Error processing job {job_id}: {e}")
            results_store[job_id] = e