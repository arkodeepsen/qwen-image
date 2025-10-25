"""
RunPod Serverless Handler for Qwen-Image
"""
import runpod
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import base64
import io
import os
from typing import Optional

# Global model instance (loaded once on cold start)
pipeline = None

def load_model():
    """Load model once during cold start"""
    global pipeline
    if pipeline is not None:
        return pipeline

    print("🚀 Loading Qwen-Image model...")

    model_name = "Qwen/Qwen-Image"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipeline = pipeline.to(device)

    print(f"✅ Model loaded on {device}")
    print(f"📊 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    return pipeline

def generate_image(job):
    """
    RunPod handler function - mirrors generate_image() from runpod_startup.sh lines 88-112
    Input format: {"input": {"prompt": "...", "width": 1024, ...}}
    Output format: {"image": "base64...", "seed": 123}
    """
    job_input = job["input"]

    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "prompt is required"}

    negative_prompt = job_input.get("negative_prompt", " ")
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)
    num_inference_steps = job_input.get("num_inference_steps", 50)
    true_cfg_scale = job_input.get("true_cfg_scale", 4.0)
    seed = job_input.get("seed", None)

    print(f"🎨 Generating: {prompt[:100]}...")

    # Load model if not already loaded
    pipe = load_model()

    # Setup generator for seed
    generator = None
    if seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

    # Generate image
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=generator
        )

    # Convert to base64
    image = result.images[0]
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    used_seed = seed if seed is not None else (generator.initial_seed() if generator else 0)

    print(f"✅ Generated successfully! Seed: {used_seed}")

    return {
        "image": img_b64,
        "seed": used_seed
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": generate_image})
