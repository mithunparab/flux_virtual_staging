import os
import torch
import random
from PIL import Image
from pathlib import Path
import traceback
import time

from flux.util import load_ae
from flux.sampling import get_schedule, denoise, unpack, prepare_kontext
from flux.trt.engine import CLIPEngine, T5Engine, TransformerEngine, SharedMemory
from config import MAX_IMAGE_SIZE, SYSTEM_PROMPT, MAX_SEED

class OfflineEngineConfig:
    def __init__(self, engine_path: str, text_maxlen: int | None = None, model_name: str | None = None):
        self.engine_path = engine_path
        self.text_maxlen = text_maxlen
        self.model_name = model_name

class StagingModel:
    def __init__(self):
        gpu_type = os.environ.get("GPU_TYPE", "H100").upper()
        self.device = torch.device("cuda")

        base_volume_path = os.environ.get("NETWORK_VOLUME_PATH", "/runpod-volume")
        autoencoder_path = Path(base_volume_path) / "models" / "flux-dev-kontext"
        if not autoencoder_path.exists():
            raise FileNotFoundError(f"FATAL: Autoencoder path '{autoencoder_path}' not found on the network volume.")

        engine_dir = Path(base_volume_path) / "engines" / gpu_type / "flux-dev-kontext"
        
        print(f"Initializing StagingModel for GPU: {gpu_type}")
        print(f"Loading engines directly from: {engine_dir}")

        if not engine_dir.exists():
            raise FileNotFoundError(f"FATAL: Engine directory '{engine_dir}' not found on the network volume.")

        transformer_precision = "fp8" if gpu_type == "H100" else "bf16"
        
        try:
            clip_engine_files = list(engine_dir.glob("clip_bf16*.plan"))
            if not clip_engine_files: raise FileNotFoundError("CLIP")
            clip_engine_path = clip_engine_files[0]

            t5_engine_files = list(engine_dir.glob("t5_bf16*.plan"))
            if not t5_engine_files: raise FileNotFoundError("T5")
            t5_engine_path = t5_engine_files[0]
            
            transformer_engine_files = list(engine_dir.glob(f"transformer_{transformer_precision}*.plan"))
            if not transformer_engine_files: raise FileNotFoundError(f"Transformer ({transformer_precision})")
            transformer_engine_path = transformer_engine_files[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(f"FATAL: Could not find a .plan file for {e} in '{engine_dir}'") from e
        
        clip_config = OfflineEngineConfig(engine_path=str(clip_engine_path), text_maxlen=77)
        t5_config = OfflineEngineConfig(engine_path=str(t5_engine_path), text_maxlen=512)
        transformer_config = OfflineEngineConfig(engine_path=str(transformer_engine_path), model_name="flux-dev-kontext")
        
        self.inference_stream = torch.cuda.Stream()
        self.context_memory = SharedMemory(1024 * 1024 * 1024 * 4)

        self.clip = CLIPEngine(clip_config, stream=self.inference_stream, context_memory=self.context_memory).to(self.device)
        self.t5 = T5Engine(t5_config, stream=self.inference_stream, context_memory=self.context_memory).to(self.device)
        self.transformer = TransformerEngine(transformer_config, stream=self.inference_stream, context_memory=self.context_memory).to(self.device)
        
        self.ae = load_ae("flux-dev-kontext", device=self.device)
        
        print("StagingModel initialized successfully using only pre-compiled engines and local files.")


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
            
            batched_images = []
            for current_seed in seeds:
                with torch.cuda.stream(self.inference_stream):
                    inp, height, width = prepare_kontext(
                        t5=self.t5, clip=self.clip, prompt=full_prompt, ae=self.ae,
                        img_cond_path=temp_img_path, 
                        seed=current_seed, device=self.device, bs=1
                    )
                    
                    timesteps = get_schedule(steps, inp["img"].shape[1], shift=True)
                    
                    inp.pop("img_cond_orig", None)
                    latents = denoise(self.transformer, timesteps=timesteps, guidance=guidance_scale, **inp)
                    
                    latents = unpack(latents.float(), height, width)
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        image = self.ae.decode(latents)
                
                torch.cuda.synchronize()

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
            traceback.print_exc()
            return e, []
        finally:
             if temp_img_path and os.path.exists(temp_img_path):
                os.remove(temp_img_path)