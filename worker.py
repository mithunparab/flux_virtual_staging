import time
from queue import Queue
from model_pipeline import StagingModel
import io         
import base64     
from config import SUPPORTED_FORMATS 

def processing_worker(model: StagingModel, job_queue: Queue, results_store: dict):
    print("Worker thread started. Waiting for jobs...")
    while True:
        try:
            job_id, job_data = job_queue.get()
            print(f"Worker received job: {job_id}")

            result_images, used_seeds = model.generate(
                prompt=job_data["prompt"],
                input_image=job_data["image"],
                seed=job_data["seed"],
                guidance_scale=job_data["guidance_scale"],
                steps=job_data["steps"],
                negative_prompt=job_data["negative_prompt"],
                num_outputs=job_data.get("num_outputs", 1), 
                aspect_ratio=job_data.get("aspect_ratio", "default"),
                super_resolution=job_data.get("super_resolution", "traditional"),
                sr_scale=job_data.get("sr_scale", 2),
                system_prompt=job_data.get("system_prompt") # ADDED
            )
            
            if isinstance(result_images, Exception):
                results_store[job_id] = result_images
                print(f"Worker stored exception for job: {job_id}")
                continue

            output_extension = job_data.get("output_extension", "jpeg").lower()
            format_info = SUPPORTED_FORMATS.get(output_extension, SUPPORTED_FORMATS['jpeg'])
            image_format = format_info['format']

            encoded_images = []
            for img in result_images:
                if image_format in ['JPEG', 'BMP'] and img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                buffered = io.BytesIO()
                img.save(buffered, format=image_format)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                encoded_images.append(img_str)
            
            results_store[job_id] = {
                "images": encoded_images,
                "seeds": used_seeds
            }
            
            print(f"Worker finished job: {job_id}")

        except Exception as e:
            print(f"Error processing job {job_id}: {e}")
            results_store[job_id] = e