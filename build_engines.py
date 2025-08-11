import os
from pathlib import Path
from huggingface_hub import snapshot_download
from flux.trt.trt_manager import TRTManager, ModuleName

if __name__ == "__main__":
    print("----------------------------------------------------")
    print("--- GPU-Powered TensorRT Engine Compilation Script ---")
    
    output_path = os.environ.get("NETWORK_VOLUME_PATH")
    if not output_path:
        raise ValueError("FATAL: NETWORK_VOLUME_PATH environment variable is not set. "
                         "Please set it to your RunPod network volume mount point (e.g., /workspace/engines).")

    gpu_type = os.environ.get("GPU_TYPE", "H100").upper()
    model_name = "black-forest-labs/FLUX.1-Kontext-dev"
    onnx_model_name = "black-forest-labs/FLUX.1-Kontext-dev-onnx"
    
    temp_cache = "./temp_model_cache"
    print(f"Downloading models to temporary cache: {temp_cache}")
    snapshot_download(repo_id=model_name, cache_dir=temp_cache, local_dir_use_symlinks=False)
    snapshot_download(repo_id=onnx_model_name, cache_dir=temp_cache, local_dir_use_symlinks=False)

    engine_dir = Path(output_path) / gpu_type
    os.makedirs(engine_dir, exist_ok=True)
    print(f"Engines will be saved to: {engine_dir}")

    transformer_precision = "fp8" if gpu_type == "H100" else "bf16"
    t5_precision = "bf16"
    print(f"Target GPU: {gpu_type}, using Transformer Precision: {transformer_precision}, T5 Precision: {t5_precision}")

    onnx_base_path = Path(temp_cache) / f"models--{onnx_model_name.replace('/', '--')}" / "snapshots"
    latest_onnx_snapshot = sorted(os.listdir(onnx_base_path))[-1]
    onnx_snapshot_path = onnx_base_path / latest_onnx_snapshot
    
    onnx_paths = {
        ModuleName.CLIP: onnx_snapshot_path / "text_encoder.onnx",
        ModuleName.T5: onnx_snapshot_path / f"t5_{t5_precision}" / "t5.onnx",
        ModuleName.TRANSFORMER: onnx_snapshot_path / f"transformer_{transformer_precision}" / "transformer.onnx",
    }
    
    for name, path in onnx_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"ONNX file for {name.name} not found at {path}")
        print(f"Found ONNX file for {name.name}: {path}")

    manager = TRTManager(trt_transformer_precision=transformer_precision, trt_t5_precision=t5_precision)
    
    manager.load_engines(
        model_name=model_name, 
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

    print("\n----------------------------------------------------")
    print(f"--- TensorRT Engines Built and Saved to {engine_dir} ---")
    print("--- You can now shut down this GPU pod. ---")
    print("----------------------------------------------------")