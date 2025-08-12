import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
from flux.trt.trt_manager import TRTManager, ModuleName

def build():
    output_path = os.environ.get("NETWORK_VOLUME_PATH")
    if not output_path:
        raise ValueError("FATAL: NETWORK_VOLUME_PATH environment variable is not set.")

    BUILD_CACHE_DIR = Path("./build_model_assets")
    
    gpu_type = os.environ.get("GPU_TYPE", "H100").upper()
    main_model_id = "black-forest-labs/FLUX.1-Kontext-dev"
    onnx_model_id = "black-forest-labs/FLUX.1-Kontext-dev-onnx"
    library_model_name = "flux-dev-kontext"

    original_cache = os.environ.get('HUGGING_FACE_HUB_CACHE')
    original_offline = os.environ.get('HF_HUB_OFFLINE')

    try:
        if not BUILD_CACHE_DIR.exists():
            print(f"\n[PHASE 1/3] Build cache not found. Downloading required assets to '{BUILD_CACHE_DIR}'...")
            os.makedirs(BUILD_CACHE_DIR, exist_ok=True)
            
            snapshot_download(
                repo_id=main_model_id,
                cache_dir=BUILD_CACHE_DIR,
                allow_patterns=["*.json", "*.txt"],
                local_dir_use_symlinks=False,
            )

            snapshot_download(
                repo_id=onnx_model_id,
                cache_dir=BUILD_CACHE_DIR,
                local_dir_use_symlinks=False
            )
            print("[SUCCESS] All required assets downloaded.")
        else:
            print(f"\n[PHASE 1/3] Found existing build cache at '{BUILD_CACHE_DIR}'. Skipping download.")

        print("\n[PHASE 2/3] Building TensorRT engines from local assets...")
        
        os.environ['HUGGING_FACE_HUB_CACHE'] = str(BUILD_CACHE_DIR)
        os.environ['HF_HUB_OFFLINE'] = '1'

        onnx_model_info = HfApi().model_info(onnx_model_id)
        onnx_repo_cache_path = BUILD_CACHE_DIR / f"models--{onnx_model_id.replace('/', '--')}" / "snapshots" / onnx_model_info.sha

        engine_dir = Path(output_path) / gpu_type
        os.makedirs(engine_dir, exist_ok=True)
        print(f"Engines will be saved to: {engine_dir}")

        transformer_precision = "fp8" if gpu_type == "H100" else "bf16"
        t5_precision = "bf16"
        
        onnx_paths_dict = {
            ModuleName.CLIP.value: str(onnx_repo_cache_path / "clip.opt" / "model.onnx"),
            ModuleName.T5.value: str(onnx_repo_cache_path / "t5.opt" / "model.onnx"),
            ModuleName.TRANSFORMER.value: str(onnx_repo_cache_path / "transformer.opt" / transformer_precision / "model.onnx"),
        }

        for name, path_str in onnx_paths_dict.items():
            if not Path(path_str).exists():
                raise FileNotFoundError(f"ONNX file for {name} not found at {path_str}")

        onnx_paths_str = ",".join([f"{key}:{value}" for key, value in onnx_paths_dict.items()])

        manager = TRTManager(
            trt_transformer_precision=transformer_precision,
            trt_t5_precision=t5_precision,
        )
        
        manager.load_engines(
            model_name=library_model_name,
            module_names={ModuleName.CLIP, ModuleName.T5, ModuleName.TRANSFORMER},
            engine_dir=str(engine_dir),
            custom_onnx_paths=onnx_paths_str,
            trt_image_height=1024,
            trt_image_width=1024,
            
        )
        print(f"[SUCCESS] TensorRT engines built and saved to {engine_dir}")

    finally:
        if original_cache is None:
            if 'HUGGING_FACE_HUB_CACHE' in os.environ:
                del os.environ['HUGGING_FACE_HUB_CACHE']
        else:
            os.environ['HUGGING_FACE_HUB_CACHE'] = original_cache

        if original_offline is None:
            if 'HF_HUB_OFFLINE' in os.environ:
                del os.environ['HF_HUB_OFFLINE']
        else:
            os.environ['HF_HUB_OFFLINE'] = original_offline
            
    print("\n[PHASE 3/3] Cleaning up temporary build assets...")
    shutil.rmtree(BUILD_CACHE_DIR)
    print(f"[SUCCESS] Deleted temporary directory: {BUILD_CACHE_DIR}")

if __name__ == "__main__":
    build()
    print("\n[COMPLETE] Build process finished successfully.")