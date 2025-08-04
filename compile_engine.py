import torch
import os
import argparse
from diffusers import FluxKontextPipeline
from diffusers.quantizers import PipelineQuantizationConfig, TorchAoConfig
import torch_tensorrt

def compile_model_to_trt(gpu_type: str, output_path: str = "./engine"):
    """
    Loads the FLUX.1 model, applies hardware-specific optimizations (FP8 for H100),
    compiles it with TensorRT, and saves the engine.
    """
    print(f"Starting compilation for GPU_TYPE: {gpu_type}")
    
    dtype = torch.bfloat16
    pipeline_kwargs = {
        "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-Kontext-dev",
        "torch_dtype": dtype,
    }

    if gpu_type.upper() == "H100":
        print("Applying H100-specific FP8 quantization...")
        quant_config = PipelineQuantizationConfig(
            quant_mapping={"transformer": TorchAoConfig("float8dq_e4m3_row")}
        )
        pipeline_kwargs["quantization_config"] = quant_config
    
    pipe = FluxKontextPipeline.from_pretrained(**pipeline_kwargs)
    pipe.to("cuda")
    print("Original pipeline loaded onto GPU.")

    transformer = pipe.transformer
    transformer.eval() 

    text_embeds = torch.randn(2, 1, 1536, dtype=dtype, device="cuda")
    pooled_embeds = torch.randn(2, 1, 1280, dtype=dtype, device="cuda")
    latents = torch.randn(2, 4, 128, 128, dtype=dtype, device="cuda") # 1024/8
    img_ids = torch.ones(2, dtype=torch.long, device="cuda")
    timestep = torch.randn(2, dtype=torch.long, device="cuda")
    
    print("Compiling transformer with TensorRT...")
    trt_transformer_ts = torch.jit.trace(transformer, (text_embeds, pooled_embeds, latents, img_ids, timestep))
    
    trt_transformer = torch_tensorrt.compile(
        trt_transformer_ts,
        inputs=[
            torch_tensorrt.Input(min_shape=(1, 1, 1536), opt_shape=(2, 1, 1536), max_shape=(4, 1, 1536), dtype=dtype),
            torch_tensorrt.Input(min_shape=(1, 1, 1280), opt_shape=(2, 1, 1280), max_shape=(4, 1, 1280), dtype=dtype),
            torch_tensorrt.Input(min_shape=(1, 4, 64, 64), opt_shape=(2, 4, 128, 128), max_shape=(4, 4, 128, 128), dtype=dtype),
            torch_tensorrt.Input(min_shape=(1,), opt_shape=(2,), max_shape=(4,), dtype=torch.long),
            torch_tensorrt.Input(min_shape=(1,), opt_shape=(2,), max_shape=(4,), dtype=torch.long),
        ],
        enabled_precisions={dtype},
        workspace_size=1 << 32, 
        truncate_long_and_double=True,
    )
    
    os.makedirs(output_path, exist_ok=True)
    
    engine_path = os.path.join(output_path, f"transformer_{gpu_type.lower()}.ts")
    torch.jit.save(trt_transformer, engine_path)
    print(f"TensorRT transformer engine saved to: {engine_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_type", type=str, required=True, choices=["H100", "L40"], help="The target GPU for compilation.")
    args = parser.parse_args()
    
    output_dir = os.path.join("./engines", args.gpu_type)
    compile_model_to_trt(gpu_type=args.gpu_type, output_path=output_dir)