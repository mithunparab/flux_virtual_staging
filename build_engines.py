import os
import torch
from flux.util import check_onnx_access_for_trt
from flux.trt.trt_manager import TRTManager, ModuleName


if __name__ == "__main__":
    print("----------------------------------------------------")
    print("--- Starting One-Time TensorRT Engine Compilation ---")
    print("This will take several minutes and only runs once during image build.")

    gpu_type = os.environ.get("GPU_TYPE", "H100").upper()
    model_name = "flux-dev-kontext"
    engine_dir = f"./engines/{gpu_type}"

    onnx_paths = check_onnx_access_for_trt(model_name, trt_transformer_precision="fp8")

    manager = TRTManager(trt_transformer_precision="fp8", trt_t5_precision="bf16")

    manager.load_engines(
        model_name=model_name,
        module_names={ModuleName.CLIP, ModuleName.T5, ModuleName.TRANSFORMER},
        engine_dir=engine_dir,
        custom_onnx_paths=onnx_paths,
        trt_image_height=1024,
        trt_image_width=1024,
        trt_static_batch=False,
        trt_static_shape=False, 
    )

    print("--- TensorRT Engines Built and Cached Successfully ---")
    print("----------------------------------------------------")