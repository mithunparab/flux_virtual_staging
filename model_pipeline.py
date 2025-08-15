import torch
import gc
from PIL import Image
from diffusers import FluxKontextPipeline
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
        width, height = image.size
        
        target_width, target_height = width, height

        if aspect_ratio == 'square':
            print("Applying center crop to create a square image...")
            crop_size = min(width, height)
            left, top = (width - crop_size) / 2, (height - crop_size) / 2
            right, bottom = (width + crop_size) / 2, (height + crop_size) / 2
            image = image.crop((left, top, right, bottom))
            target_width, target_height = MAX_IMAGE_SIZE, MAX_IMAGE_SIZE
            
        elif aspect_ratio == 'portrait':
            print("Applying crop for portrait (2:3) aspect ratio...")
            target_ratio = 2 / 3
            current_ratio = width / height
            if current_ratio > target_ratio: 
                new_width = int(target_ratio * height)
                left, top = (width - new_width) / 2, 0
                right, bottom = left + new_width, height
            else: 
                new_height = int(width / target_ratio)
                left, top = 0, (height - new_height) / 2
                right, bottom = width, top + new_height
            image = image.crop((left, top, right, bottom))
            target_height = MAX_IMAGE_SIZE
            target_width = int(MAX_IMAGE_SIZE * target_ratio)

        elif aspect_ratio == 'landscape':
            print("Applying crop for landscape (3:2) aspect ratio...")
            target_ratio = 3 / 2
            current_ratio = width / height
            if current_ratio > target_ratio: 
                new_width = int(target_ratio * height)
                left, top = (width - new_width) / 2, 0
                right, bottom = left + new_width, height
            else: 
                new_height = int(width / target_ratio)
                left, top = 0, (height - new_height) / 2
                right, bottom = width, top + new_height
            image = image.crop((left, top, right, bottom))
            target_width = MAX_IMAGE_SIZE
            target_height = int(MAX_IMAGE_SIZE / target_ratio)
        
        if max(image.size) > MAX_IMAGE_SIZE:
             print(f"Resizing image from {image.size} to fit within ({target_width}, {target_height})")
             image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
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

    def generate(self, prompt: str, input_image: Image.Image, seed: int, guidance_scale: float, steps: int, negative_prompt: str, num_outputs: int, aspect_ratio: str, super_resolution: str, sr_scale: int):
        full_prompt = f"{SYSTEM_PROMPT}{prompt}"
        print(f"Using full prompt: {full_prompt}")
        print(f"Using negative prompt: {negative_prompt}")
        print(f"Generating {num_outputs} image(s)...")

        with self._inference_context():
            try:
                with torch.no_grad():
                    input_image = input_image.convert("RGB")
                    input_image = self._resize_image(input_image, aspect_ratio)

                    images_result = self.pipe(
                        image=input_image,
                        prompt=full_prompt,
                        negative_prompt=negative_prompt,
                        guidance_scale=guidance_scale,
                        width=input_image.size[0],
                        height=input_image.size[1],
                        num_inference_steps=steps,
                        num_images_per_prompt=num_outputs,
                        generator=torch.Generator("cuda").manual_seed(seed),
                    ).images
                
                if super_resolution == "traditional" and sr_scale > 1:
                    print(f"Applying traditional super resolution with scale x{sr_scale}")
                    final_images = []
                    for img in images_result:
                        new_width = img.width * sr_scale
                        new_height = img.height * sr_scale
                        final_images.append(img.resize((new_width, new_height), Image.Resampling.LANCZOS))
                    return final_images

                return images_result

            except Exception as e:
                print(f"An error occurred during inference: {e}")
                import traceback
                traceback.print_exc()
                return e