import torch
import gc
from PIL import Image
from diffusers import FluxKontextPipeline
from contextlib import contextmanager

from config import MODEL_ID, MAX_IMAGE_SIZE, SYSTEM_PROMPT

class StagingModel:
    def __init__(self):
        print("Loading FLUX.1 pipeline...")
        self.pipe = FluxKontextPipeline.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.bfloat16
        )
        print("Moving pipeline to GPU...")
        self.pipe.to("cuda")
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("xformers memory-efficient attention enabled.")
        except ImportError:
            print("[Error] xformers not installed. For better performance, install it with 'pip install xformers'")
        except AttributeError:
            print("[Error] Your version of `diffusers` is too old to support `enable_xformers_memory_efficient_attention()`.")
            print("For better performance, please upgrade with: `pip install --upgrade diffusers`")
        except Exception as e:
            print(f"An unexpected error occurred while enabling xformers: {e}")
        
        print("Pipeline loaded and configured.")

    def _resize_image(self, image: Image.Image) -> Image.Image:
        if max(image.size) > MAX_IMAGE_SIZE:
            image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
            print(f"Resized image to {image.size} to fit within {MAX_IMAGE_SIZE}x{MAX_IMAGE_SIZE}")
        return image

    @contextmanager
    def _inference_context(self):
        torch.cuda.empty_cache()
        gc.collect()
        try:
            yield
        finally:
            torch.cuda.empty_cache()
            gc.collect()
            print("GPU memory cleaned up after inference.")

    def generate(self, prompt: str, input_image: Image.Image, seed: int, guidance_scale: float, steps: int, negative_prompt: str):
        full_prompt = f"{SYSTEM_PROMPT}{prompt}"
        print(f"Using full prompt: {full_prompt}")
        print(f"Using negative prompt: {negative_prompt}")
        
        with self._inference_context():
            try:
                with torch.no_grad():
                    input_image = input_image.convert("RGB")
                    input_image = self._resize_image(input_image)
                    
                    image_result = self.pipe(
                        image=input_image, 
                        prompt=full_prompt,
                        negative_prompt=negative_prompt,
                        guidance_scale=guidance_scale,
                        width=input_image.size[0],
                        height=input_image.size[1],
                        num_inference_steps=steps,
                        generator=torch.Generator("cuda").manual_seed(seed),
                    ).images[0]
                
                return image_result

            except Exception as e:
                print(f"An error occurred during inference: {e}")
                import traceback
                traceback.print_exc()
                return e