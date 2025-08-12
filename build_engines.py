import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from flux.trt.trt_manager import TRTManager, ModuleName

if __name__ == "__main__":
    print("---------------------------------------------------------")
    print("--- Optimized TensorRT Engine Compilation Script      ---")
    print("--- Downloads ONLY necessary ONNX & config files.     ---")
    
    output_path = os.environ.get("NETWORK_VOLUME_PATH")
    if not output_path:
        raise ValueError("FATAL: NETWORK_VOLUME_PATH environment variable is not set.")

    BUILD_CACHE_DIR = Path("./build_model_assets")
    
    gpu_type = os.environ.get("GPU_TYPE", "H100").upper()
    main_model_id = "black-forest-labs/FLUX.1-Kontext-dev"
    onnx_model_id = "black-forest-labs/FLUX.1-Kontext-dev-onnx"
    
    print(f"\n[PHASE 1/4] Downloading required assets to '{BUILD_CACHE_DIR}'...")
    
    local_main_model_configs_path = BUILD_CACHE_DIR / "main_model_configs"
    local_onnx_model_path = BUILD_CACHE_DIR / "onnx_model"

    try:
        print(f"Downloading model configurations from '{main_model_id}'...")
        snapshot_download(
            repo_id=main_model_id,
            local_dir=local_main_model_configs_path,
            local_dir_use_symlinks=False,
            allow_patterns=["*.json", "*.txt"] 
        )

        print(f"Downloading ONNX models from '{onnx_model_id}'...")
        snapshot_download(
            repo_id=onnx_model_id,
            local_dir=local_onnx_model_path,
            local_dir_use_symlinks=False
        )
        print("[SUCCESS] All required assets downloaded efficiently.")

        print("\n[PHASE 2/4] Building TensorRT engines from local assets...")
        engine_dir = Path(output_path) / gpu_type
        os.makedirs(engine_dir, exist_ok=True)
        print(f"Engines will be saved to: {engine_dir}")

        transformer_precision = "fp8" if gpu_type == "H100" else "bf16"
        t5_precision = "bf16"
        print(f"Target GPU: {gpu_type}, Transformer Precision: {transformer_precision}, T5 Precision: {t5_precision}")

        onnx_paths = {
            ModuleName.CLIP: local_onnx_model_path / "text_encoder.onnx",
            ModuleName.T5: local_onnx_model_path / f"t5_{t5_precision}" / "t5.onnx",
            ModuleName.TRANSFORMER: local_onnx_model_path / f"transformer_{transformer_precision}" / "transformer.onnx",
        }

        for name, path in onnx_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"ONNX file for {name.name} not found at {path}")

        manager = TRTManager(trt_transformer_precision=transformer_precision, trt_t5_precision=t5_precision)
        
        manager.load_engines(
            model_name=str(local_main_model_configs_path),
            module_names={ModuleName.CLIP, ModuleName.T5, ModuleName.TRANSFORMER},
            engine_dir=str(engine_dir),
            custom_onnx_paths=onnx_paths,
            trt_min_text_len=512,
            trt_opt_text_len=2048,
            trt_max_text_len=8192,
            trt_image_height=1024,
            trt_image_width=1024,
            trt_static_batch=False,
            trt_static_shape=False,
        )
        print(f"[SUCCESS] TensorRT engines built and saved to {engine_dir}")

    finally:
        print("\n[PHASE 3/4] Cleaning up temporary build assets...")
        if BUILD_CACHE_DIR.exists():
            shutil.rmtree(BUILD_CACHE_DIR)
            print(f"[SUCCESS] Deleted temporary directory: {BUILD_CACHE_DIR}")

    print("\n[PHASE 4/4] Process Complete.")
    print("---------------------------------------------------------")