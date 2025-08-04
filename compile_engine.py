import torch
import os
import argparse
from diffusers import FluxKontextPipeline
import torch_tensorrt
from torch.export import export

def compile_model_to_trt(gpu_type: str, output_path: str):
    print(f"Starting compilation for black-forest-labs/FLUX.1-Kontext-dev.")

    dtype = torch.bfloat16
    model_id = "black-forest-labs/FLUX.1-Kontext-dev"
    DEVICE = "cuda:0"

    pipeline_kwargs = {
        "pretrained_model_name_or_path": model_id,
        "torch_dtype": dtype,
    }
    
    pipe = FluxKontextPipeline.from_pretrained(**pipeline_kwargs)
    pipe.to(DEVICE)
    print("Standard BF16 Kontext pipeline loaded on GPU.")

    transformer = pipe.transformer
    transformer.eval()

    IN_CHANNELS = transformer.x_embedder.in_channels
    D_EMBED = transformer.context_embedder.in_features
    D_COND = transformer.pooled_projection_embedder.in_features
    
    print(f"Determined model dimensions: in_channels={IN_CHANNELS}, d_embed={D_EMBED}, d_cond={D_COND}")

    VAE_DOWNSAMPLE_FACTOR = 8
    height = 1024
    width = 1024
    batch_size = 2
    
    latent_height = height // VAE_DOWNSAMPLE_FACTOR
    latent_width = width // VAE_DOWNSAMPLE_FACTOR
    
    BATCH = torch.export.Dim("batch", min=1, max=batch_size)
    
    dummy_inputs = {
        "hidden_states": torch.randn(batch_size, IN_CHANNELS, latent_height, latent_width, dtype=dtype, device=DEVICE),
        "text_embeds": torch.randn(batch_size, 1, D_EMBED, dtype=dtype, device=DEVICE),
        "pooled_projections": torch.randn(batch_size, 1, D_COND, dtype=dtype, device=DEVICE),
        "img_ids": torch.ones(batch_size, dtype=torch.long, device=DEVICE),
        "timestep": torch.randint(0, 1000, (batch_size,), dtype=torch.long, device=DEVICE),
    }

    dynamic_shapes = {
        "hidden_states": {0: BATCH},
        "text_embeds": {0: BATCH},
        "pooled_projections": {0: BATCH},
        "img_ids": {0: BATCH},
        "timestep": {0: BATCH},
    }

    print("Exporting the transformer backbone using torch.export...")
    ep = export(transformer, args=(), kwargs=dummy_inputs, dynamic_shapes=dynamic_shapes)
    
    print("Compiling exported program with TensorRT using dynamo backend...")
    
    enabled_precisions = {dtype}
    if gpu_type.upper() == "H100":
        print("Enabling FP8 precision for TensorRT compilation.")
        enabled_precisions.add(torch.float8_e4m3fn)

    trt_gm = torch_tensorrt.dynamo.compile(
        ep,
        inputs=dummy_inputs,
        enabled_precisions=enabled_precisions,
        truncate_double=True,
        min_block_size=5,
        use_fp32_acc=True,
        use_explicit_typing=True,
    )

    print("Saving compiled TensorRT engine...")
    os.makedirs(output_path, exist_ok=True)
    engine_path = os.path.join(output_path, f"transformer_{gpu_type.lower()}.pt")
    
    # Trace the final compiled graph for a stable, serializable artifact
    traced_trt_model = torch.jit.trace(trt_gm, example_kw_args=dummy_inputs)
    torch.jit.save(traced_trt_model, engine_path)
    
    print(f"TensorRT transformer engine saved to: {engine_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_type", type=str, required=True, choices=["H100", "L40"])
    parser.add_argument("--output_dir", type=str, default="./engines", help="Directory to save the compiled engine.")
    args = parser.parse_args()
    
    output_dir = os.path.join(args.output_dir, args.gpu_type)
    compile_model_to_trt(gpu_type=args.gpu_type, output_path=output_dir)