import torch
import os
import random
from PIL import Image

from flux.util import load_ae, check_onnx_access_for_trt
from flux.trt.trt_manager import TRTManager, ModuleName
from flux.sampling import get_schedule, denoise, unpack, prepare_kontext

from config import MAX_IMAGE_SIZE, SYSTEM_PROMPT, MAX_SEED

class StagingModel:
    def __init__(self):
        """
        Initializes the model for both local and production environments.
        - In production (Docker), it loads pre-compiled TensorRT engines.
        - Locally, it builds and caches the engines on the first run.
        """
        gpu_type = os.environ.get("GPU_TYPE", "H100").upper() 
        self.device = torch.device("cuda")
        self.model_name = "flux-dev-kontext"
        
        transformer_precision = "fp8" if gpu_type == "H100" else "bf16"
        print(f"Initializing StagingModel for GPU: {gpu_type}, using Transformer Precision: {transformer_precision}")

        onnx_paths = check_onnx_access_for_trt(self.model_name, trt_transformer_precision=transformer_precision)

        manager = TRTManager(trt_transformer_precision=transformer_precision, trt_t5_precision="bf16")

        self.engines = manager.load_engines(
            model_name=self.model_name,
            module_names={ModuleName.CLIP, ModuleName.T5, ModuleName.TRANSFORMER},
            engine_dir=f"./engines/{gpu_type}",
            custom_onnx_paths=onnx_paths,
            trt_image_height=1024,
            trt_image_width=1024,
            trt_static_batch=False,
            trt_static_shape=False,
        )

        self.clip = self.engines[ModuleName.CLIP].to(self.device)
        self.t5 = self.engines[ModuleName.T5].to(self.device)
        self.transformer = self.engines[ModuleName.TRANSFORMER].to(self.device)

        self.inference_stream = torch.cuda.Stream()
        self.clip.stream = self.inference_stream
        self.t5.stream = self.inference_stream
        self.transformer.stream = self.inference_stream

        self.ae = load_ae(self.model_name, device=self.device)

        print("StagingModel initialized successfully with all TensorRT engines.")

    def _resize_image(self, image: Image.Image, aspect_ratio: str) -> Image.Image:
        if aspect_ratio == 'square':
            width, height = image.size
            crop_size = min(width, height)
            left = (width - crop_size) / 2
            top = (height - crop_size) / 2
            right = (width + crop_size) / 2
            bottom = (height + crop_size) / 2
            return image.crop((left, top, right, bottom))
        return image
        
    @torch.inference_mode()
    def generate(
        self, prompt: str, input_image: Image.Image, seed: int, guidance_scale: float,
        steps: int, negative_prompt: str, aspect_ratio: str, super_resolution: str,
        sr_scale: int, num_outputs: int = 1, system_prompt: str | None = None
    ):
        final_system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        full_prompt = f"{final_system_prompt}{prompt}"
        
        temp_img_path = None
        try:
            temp_img_path = f"/tmp/input_image_{random.randint(1000, 99999)}.png"
            processed_image = self._resize_image(input_image.convert("RGB"), aspect_ratio)
            processed_image.save(temp_img_path)

            if seed == -1:
                seeds = [random.randint(0, MAX_SEED) for _ in range(num_outputs)]
            else:
                seeds = [seed + i for i in range(num_outputs)]
            
            print(f"Generating {num_outputs} image(s) with seeds: {seeds}")

            batched_images = []
            for current_seed in seeds:
                with torch.cuda.stream(self.inference_stream):
                    inp, height, width = prepare_kontext(
                        t5=self.t5, clip=self.clip, prompt=full_prompt, ae=self.ae,
                        img_cond_path=temp_img_path, seed=current_seed, device=self.device, bs=1
                    )
                    
                    inp.pop("img_cond_orig", None)
                    timesteps = get_schedule(steps, inp["img"].shape[1], shift=True)
                    latents = denoise(self.transformer, timesteps=timesteps, guidance=guidance_scale, **inp)
                    latents = unpack(latents.float(), height, width)
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        image = self.ae.decode(latents)
                
                image = (image[0].clamp(-1, 1) + 1) / 2
                image = (image.permute(1, 2, 0) * 255).byte().cpu().numpy()
                pil_image = Image.fromarray(image)
                batched_images.append(pil_image)

            final_images = []
            if super_resolution == "traditional" and sr_scale > 1:
                for img in batched_images:
                    new_width = img.width * sr_scale
                    new_height = img.height * sr_scale
                    final_images.append(img.resize((new_width, new_height), Image.Resampling.LANCZOS))
            else:
                final_images = batched_images

            return final_images, seeds

        except Exception as e:
            print(f"An error occurred during inference: {e}")
            import traceback
            traceback.print_exc()
            return e, []
        finally:
            if temp_img_path and os.path.exists(temp_img_path):
                os.remove(temp_img_path)