import torch
import os
import random
from PIL import Image

from flux.util import load_ae
from flux.trt.engine import CLIPEngine, T5Engine, TransformerEngine
from flux.sampling import get_schedule, denoise, unpack, prepare_kontext

from config import MAX_IMAGE_SIZE, SYSTEM_PROMPT, MAX_SEED

class StagingModel:
    def __init__(self):
        """
        Initializes the model by loading pre-compiled TensorRT engines.
        This assumes the engines were built during the Docker image creation process.
        """
        gpu_type = os.environ.get("GPU_TYPE", "H100").upper()
        self.device = torch.device("cuda")
        self.model_name = "flux-dev-kontext"
        engine_dir = f"./engines/{gpu_type}"
        
        print(f"Initializing StagingModel for {gpu_type} from pre-compiled engines in: {engine_dir}")

        clip_plan_path = os.path.join(engine_dir, "clip.plan")
        t5_plan_path = os.path.join(engine_dir, "t5.plan")
        transformer_plan_path = os.path.join(engine_dir, "transformer.plan")

        for path in [clip_plan_path, t5_plan_path, transformer_plan_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"FATAL: Required TensorRT engine not found at '{path}'. "
                    "Please ensure the Docker build process completes successfully."
                )
        
        print("Creating a dedicated non-default CUDA stream for TensorRT execution.")
        self.inference_stream = torch.cuda.Stream()
        
        from flux.trt.trt_config import ClipConfig, T5Config, TransformerConfig
        
        self.clip = CLIPEngine(ClipConfig.from_args(self.model_name, engine_dir=engine_dir), stream=self.inference_stream).to(self.device)
        self.t5 = T5Engine(T5Config.from_args(self.model_name, engine_dir=engine_dir), stream=self.inference_stream).to(self.device)
        self.transformer = TransformerEngine(TransformerConfig.from_args(self.model_name, engine_dir=engine_dir), stream=self.inference_stream).to(self.device)

        self.ae = load_ae(self.model_name, device=self.device)

        print("StagingModel initialized successfully with all TensorRT engines on a dedicated stream.")

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
                # All operations within this loop will now use the dedicated stream.
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