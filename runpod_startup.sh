#!/bin/bash
set -e

echo "🚀 Qwen-Image API Setup"
echo "========================"

# Clean disk space before model download
echo "🧹 Cleaning disk space..."
rm -rf /workspace/.cache/huggingface/hub/models--* 2>/dev/null || true
rm -rf /root/.cache/huggingface/* 2>/dev/null || true
rm -rf /tmp/* 2>/dev/null || true
pip cache purge 2>/dev/null || true
echo "📊 Disk space: $(df -h /workspace | tail -1 | awk '{print $4}') available"

# Install dependencies (official Qwen-Image setup)
echo "📦 Installing dependencies..."
python3 -m pip install --upgrade pip

# Core dependencies
python3 -m pip install git+https://github.com/huggingface/diffusers
python3 -m pip install transformers accelerate safetensors
python3 -m pip install hf-transfer

# API dependencies
python3 -m pip install fastapi uvicorn pillow requests

# Clean cache after install
python3 -m pip cache purge
echo "✅ Dependencies installed!"
echo "📊 PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "📊 Diffusers: $(python3 -c 'import diffusers; print(diffusers.__version__)')"
echo "📊 Transformers: $(python3 -c 'import transformers; print(transformers.__version__)')"
echo "📊 Accelerate: $(python3 -c 'import accelerate; print(accelerate.__version__)')"
echo "📊 HF Transfer: $(python3 -c 'import hf_transfer; print(hf_transfer.__version__)')"

# Create minimal API server
echo "🔧 Creating API server..."
cat > /workspace/qwen_api.py << 'EOF'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import DiffusionPipeline
import torch
import uvicorn
from PIL import Image
import base64
import io
import asyncio
import uuid
from typing import Optional

app = FastAPI(title="Qwen-Image API")
pipeline = None
jobs = {}  # In-memory job store

@app.on_event("startup")
async def startup():
    global pipeline
    print("🚀 Loading Qwen-Image model...")
    
    model_name = "Qwen/Qwen-Image"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipeline = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipeline = pipeline.to(device)
    
    print(f"✅ Model loaded on {device}")
    print(f"📊 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = " "
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 50
    true_cfg_scale: float = 4.0
    seed: Optional[int] = None

class GenerateResponse(BaseModel):
    image: str
    seed: int

class JobStatus(BaseModel):
    job_id: str
    status: str
    detail: Optional[str] = None

def generate_image(request: GenerateRequest):
    """Synchronous image generation"""
    generator = None
    if request.seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(request.seed)
    
    with torch.inference_mode():
        result = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            true_cfg_scale=request.true_cfg_scale,
            generator=generator
        )
    
    image = result.images[0]
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    used_seed = request.seed if request.seed is not None else (generator.initial_seed() if generator else 0)
    
    return img_b64, used_seed

async def run_generation_job(job_id: str, request: GenerateRequest):
    """Async job runner"""
    jobs[job_id] = {"status": "running"}
    try:
        print(f"🎨 Job {job_id}: Generating {request.prompt[:50]}...")
        img_b64, used_seed = generate_image(request)
        jobs[job_id] = {"status": "done", "image": img_b64, "seed": used_seed}
        print(f"✅ Job {job_id}: Complete!")
    except Exception as e:
        jobs[job_id] = {"status": "error", "detail": str(e)}
        print(f"❌ Job {job_id}: Failed - {e}")

@app.get("/")
async def root():
    return {"message": "Qwen-Image API", "docs": "/docs"}

@app.get("/health")
async def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "Qwen/Qwen-Image"}

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Synchronous generation (use for small images/steps)"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    print(f"🎨 Generating: {request.prompt[:100]}...")
    img_b64, used_seed = generate_image(request)
    print("✅ Generated successfully!")
    return GenerateResponse(image=img_b64, seed=used_seed)

@app.post("/generate_async", response_model=JobStatus)
async def generate_async(request: GenerateRequest):
    """Async generation (recommended for large images/many steps)"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued"}
    asyncio.create_task(run_generation_job(job_id, request))
    return JobStatus(job_id=job_id, status="queued")

@app.get("/status/{job_id}", response_model=JobStatus)
async def job_status(job_id: str):
    """Check job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    entry = jobs[job_id]
    return JobStatus(job_id=job_id, status=entry["status"], detail=entry.get("detail"))

@app.get("/result/{job_id}", response_model=GenerateResponse)
async def job_result(job_id: str):
    """Get job result"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    entry = jobs[job_id]
    if entry["status"] != "done":
        raise HTTPException(status_code=202, detail=f"Job status: {entry['status']}")
    return GenerateResponse(image=entry["image"], seed=entry["seed"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

echo "✅ Setup complete!"
echo "🚀 Starting API server..."
python3 /workspace/qwen_api.py
