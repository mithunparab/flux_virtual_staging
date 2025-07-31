import torch
import gc
from PIL import Image
from diffusers import FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from contextlib import contextmanager

from config import MODEL_ID, MAX_IMAGE_SIZE, SYSTEM_PROMPT

class StagingModel:
    def __init__(self):
        self.pipe = self._load_model()
        print("StagingModel initialized and pipeline loaded onto GPU.")

    def _load_model(self):
        """
        This function loads the model into memory.
        """
        print("Loading FLUX.1 pipeline...")
        pipe = FluxKontextPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16
        )
        print("Moving pipeline to GPU...")
        pipe.to("cuda")

        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("xformers memory-efficient attention enabled.")
        except Exception as e:
            print(f"Could not enable xformers memory-efficient attention: {e}")

        print("Pipeline loaded and configured successfully.")
        
        return pipe

    def _resize_image(self, image: Image.Image, aspect_ratio: str) -> Image.Image:
        if aspect_ratio == 'square':
            print("Applying center crop to create a square image...")
            width, height = image.size
            
            crop_size = min(width, height)
            
            left = (width - crop_size) / 2
            top = (height - crop_size) / 2
            right = (width + crop_size) / 2
            bottom = (height + crop_size) / 2

            image = image.crop((left, top, right, bottom))
            
            if image.width > MAX_IMAGE_SIZE:
                 image = image.resize((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
            print(f"Cropped and resized image to square: {image.size}")
        else:  
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

    def generate(self, prompt: str, input_image: Image.Image, seed: int, guidance_scale: float, steps: int, negative_prompt: str, aspect_ratio: str, super_resolution: str, sr_scale: int):
        full_prompt = f"{SYSTEM_PROMPT}{prompt}"
        print(f"Using full prompt: {full_prompt}")
        print(f"Using negative prompt: {negative_prompt}")

        with self._inference_context():
            try:
                with torch.no_grad():
                    input_image = input_image.convert("RGB")
                    input_image = self._resize_image(input_image, aspect_ratio)

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
                
                if super_resolution == "traditional" and sr_scale > 1:
                    print(f"Applying traditional super resolution with scale x{sr_scale}")
                    new_width = image_result.width * sr_scale
                    new_height = image_result.height * sr_scale
                    image_result = image_result.resize((new_width, new_height), Image.Resampling.LANCZOS)

                return image_result

            except Exception as e:
                print(f"An error occurred during inference: {e}")
                import traceback
                traceback.print_exc()
                return e