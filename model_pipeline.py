import torch
import gc
import os
from PIL import Image
from diffusers import FluxKontextPipeline, DiffusionPipeline, TorchAoConfig
from diffusers.quantizers import PipelineQuantizationConfig
import random

from config import MODEL_ID, MAX_IMAGE_SIZE, SYSTEM_PROMPT, MAX_SEED

class StagingModel:
    def __init__(self):
        gpu_type = os.environ.get("GPU_TYPE", "H100").upper()
        print(f"Initializing StagingModel for {gpu_type} using pre-compiled TensorRT engine.")
        self.pipe = self._load_model_from_engine(gpu_type)
        print("StagingModel initialized and TensorRT pipeline loaded onto GPU.")

    def _load_model_from_engine(self, gpu_type: str):
        """
        Loads a pipeline with a pre-compiled TensorRT engine for the transformer.
        """
        engine_path = f"./engines/{gpu_type}/transformer_{gpu_type.lower()}.ts"
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"FATAL: Compiled engine not found at {engine_path}. Please run compile_engine.py first.")

        print(f"Loading base pipeline structure...")
        pipe = FluxKontextPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True 
        )

        print(f"Loading compiled TensorRT engine from: {engine_path}")
        trt_transformer = torch.jit.load(engine_path)
        
        pipe.transformer = trt_transformer
        pipe.to("cuda")
        
        print("Pipeline with TensorRT engine is ready.")
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
        

    def generate(
        self,
        prompt: str,
        input_image: Image.Image,
        seed: int,
        guidance_scale: float,
        steps: int,
        negative_prompt: str,
        aspect_ratio: str,
        super_resolution: str,
        sr_scale: int,
        num_outputs: int = 1
    ):
        full_prompt = f"{SYSTEM_PROMPT}{prompt}"
        print(f"Using full prompt: {full_prompt}")
        print(f"Using negative prompt: {negative_prompt}")
        
        try:
            input_image = input_image.convert("RGB")
            input_image = self._resize_image(input_image, aspect_ratio)

            if seed != -1:
                seeds = [seed + i for i in range(num_outputs)]
            else:
                seeds = [random.randint(0, MAX_SEED) for _ in range(num_outputs)]
            
            print(f"Generating a batch of {num_outputs} with seeds: {seeds}")

            generators = [torch.Generator("cuda").manual_seed(s) for s in seeds]

            with torch.no_grad():
                batched_images = self.pipe(
                    image=input_image,
                    prompt=[full_prompt] * num_outputs,
                    negative_prompt=[negative_prompt] * num_outputs,
                    guidance_scale=guidance_scale,
                    width=input_image.size[0],
                    height=input_image.size[1],
                    num_inference_steps=steps,
                    generator=generators,
                ).images

            final_images = []
            if super_resolution == "traditional" and sr_scale > 1:
                print(f"Applying traditional super resolution with scale x{sr_scale} to {len(batched_images)} images")
                for img in batched_images:
                    new_width = img.width * sr_scale
                    new_height = img.height * sr_scale
                    final_images.append(
                        img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    )
            else:
                final_images = batched_images

            return final_images, seeds

        except Exception as e:
            print(f"An error occurred during inference: {e}")
            import traceback
            traceback.print_exc()
            return e, []
