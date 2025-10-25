# Qwen-Image Fast-API

[![RunPod](https://api.runpod.io/badge/arkodeepsen/qwen-image)](https://console.runpod.io/hub/arkodeepsen/qwen-image)

[![One-Click Pod Deployment](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/685b44aed6fc50d169003af4_banner-runpod.webp)](https://console.runpod.io/deploy?template=wqf5o3topx&ref=az0kmnor)

A production-ready RunPod serverless endpoint for Alibaba's Qwen-Image model - a powerful text-to-image generation model with superior text rendering capabilities in both English and Chinese.

## Features

- **Official Qwen-Image Model** - 20B MMDiT image foundation model
- **GPU Optimized** - Runs on A100 80GB, H100 PCIe, H100 HBM3, H100 NVL, and high-end workstation GPUs
- **Auto-scaling** - Scales to 0 when idle to save costs
- **Network Volume Storage** - Model cached persistently across all workers
- **Fast Cold Starts** - Optimized Docker image with pre-installed dependencies

## Model Specifications

- **Model**: `Qwen/Qwen-Image` (20B parameters)
- **Recommended VRAM**: 80GB (A100/H100 recommended)
- **Precision**: bfloat16 (CUDA) / float32 (CPU)
- **Default Resolution**: 1024x1024
- **Text Rendering**: Exceptional quality for both English and Chinese text
- **License**: Apache 2.0

## API Usage

### Input Format

```json
{
  "input": {
    "prompt": "Your image description here",
    "negative_prompt": " ",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "true_cfg_scale": 4.0,
    "seed": null
  }
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Description of the image to generate |
| `negative_prompt` | string | `" "` | What to avoid in the image |
| `width` | integer | `1024` | Image width (pixels) |
| `height` | integer | `1024` | Image height (pixels) |
| `num_inference_steps` | integer | `50` | Number of denoising steps (higher = better quality, slower) |
| `true_cfg_scale` | float | `4.0` | Classifier-free guidance scale |
| `seed` | integer | `null` | Random seed for reproducibility (optional) |

### Output Format

```json
{
  "image": "base64_encoded_png_data",
  "seed": 12345
}
```

### Example Request (Python)

```python
import runpod
import base64
from PIL import Image
import io

runpod.api_key = "your_api_key_here"

endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

request = {
    "input": {
        "prompt": "A serene mountain landscape with Chinese calligraphy 'Harmony'",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 50,
        "seed": 42
    }
}

run = endpoint.run_sync(request)

# Decode and save image
img_data = base64.b64decode(run['image'])
image = Image.open(io.BytesIO(img_data))
image.save('output.png')

print(f"Generated with seed: {run['seed']}")
```

### Example Request (cURL)

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A futuristic cityscape at sunset",
      "width": 1024,
      "height": 1024,
      "num_inference_steps": 50
    }
  }'
```

## Deployment Configuration

The template is configured with optimal settings in `runpod.toml`:

- **GPU Types**: A100 80GB PCIe, H100 PCIe, H100 HBM3, H100 NVL, RTX 6000 Blackwell, RTX 6000 Blackwell Workstation, RTX Pro 6000 Max-Q Workstation
- **Recommended VRAM**: 80GB
- **Container Disk**: 5GB (code + dependencies)
- **Network Volume**: 100GB (persistent model storage)
- **Workers**: 0-3 (auto-scaling)
- **Timeout**: 600 seconds per job

## Performance

- **Cold Start**: ~60-120 seconds (model download on first run)
- **Warm Inference**: ~20-40 seconds (depends on steps and resolution)
- **Memory Usage**: ~40-60GB VRAM for 1024x1024 images (20B parameter model)

## Tips for Best Results

1. **Prompt Quality**: Be specific and descriptive
2. **Steps**: 30-50 steps for good quality, 50-100 for best quality
3. **CFG Scale**: 3.5-5.0 works well for most prompts
4. **Text Rendering**: Qwen-Image excels at rendering text - great for logos, signs, and calligraphy
5. **Seed**: Use the same seed to reproduce images

## License

This endpoint uses the Qwen-Image model licensed under Apache 2.0. For more information, visit the [official Qwen-Image repository](https://github.com/QwenLM/Qwen-Image).

## Support

For issues or questions about this RunPod template, please open an issue on the GitHub repository.
