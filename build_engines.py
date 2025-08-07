import os
from flux.util import check_onnx_access_for_trt
from flux.trt.trt_manager import TRTManager, ModuleName

if __name__ == "__main__":
    print("----------------------------------------------------")
    print("--- Starting One-Time TensorRT Engine Compilation ---")
    
    gpu_type = os.environ.get("GPU_TYPE", "H100").upper()
    model_name = "black-forest-labs/FLUX.1-Kontext-dev"
    engine_dir = f"./engines/{gpu_type}"

    transformer_precision = "fp8" if gpu_type == "H100" else "bf16"
    print(f"Target GPU: {gpu_type}, using Transformer Precision: {transformer_precision}")

    onnx_paths = check_onnx_access_for_trt(model_name, trt_transformer_precision=transformer_precision)

    manager = TRTManager(trt_transformer_precision=transformer_precision, trt_t5_precision="bf16")

    manager.load_engines(
        model_name=model_name,
        module_names={ModuleName.CLIP, ModuleName.T5, ModuleName.TRANSFORMER},
        engine_dir=engine_dir,
        custom_onnx_paths=onnx_paths,
        trt_min_text_len=512,
        trt_opt_text_len=2048,
        trt_max_text_len=8192,
        trt_image_height=1024,
        trt_image_width=1024,
        trt_static_batch=False,
        trt_static_shape=False,
    )

    print("--- TensorRT Engines Built and Cached Successfully in Docker Image ---")
    print("----------------------------------------------------")