# Runpod Cached Model Worker

A Runpod Serverless worker that serves a Hugging Face text generation model using the [model caching](https://docs.runpod.io/serverless/endpoints/manage-endpoints#model-caching) feature. The model is pre-downloaded to a network volume and loaded at startup in offline mode, eliminating cold-start downloads.

Defaults to [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) but works with any Hugging Face text generation model available through the model caching feature.

## How it works

1. Runpod's model caching feature downloads the model to `/runpod-volume/huggingface-cache/hub/` before the worker starts.
2. The handler resolves the local snapshot path from the cache directory.
3. The model and tokenizer are loaded once at startup in offline mode (`HF_HUB_OFFLINE=1`).
4. Incoming requests are processed using a `transformers` text generation pipeline.

## Files

```
handler.py           # Serverless handler with model loading and inference
Dockerfile           # Container image based on runpod/pytorch:2.4.0
requirements.txt     # Python dependencies
build-and-push.sh    # Build and push Docker image to Docker Hub
```

## Deploy

1. Go to [Serverless > New Endpoint](https://www.console.runpod.io/serverless) in the Runpod console.
2. Under **Container Image**, enter a built image (e.g., `your-username/cached-model-worker:latest`), or use **Import Git Repository** to build directly from this repo.
3. Select a GPU with at least 16 GB VRAM.
4. Under the **Model** section, enter the model name: `microsoft/Phi-3-mini-4k-instruct`.
5. Set container disk to at least 20 GB.
6. Select **Deploy Endpoint**.

## Build and push

Using the included script:

```bash
chmod +x build-and-push.sh
./build-and-push.sh your-dockerhub-username
```

Or manually:

```bash
docker build -t your-username/cached-model-worker:latest .
docker push your-username/cached-model-worker:latest
```

Or with [Depot](https://depot.dev) for faster cloud builds:

```bash
depot build -t your-username/cached-model-worker:latest . --platform linux/amd64 --push
```

## Request format

```json
{
  "input": {
    "prompt": "What is the capital of France?",
    "max_tokens": 256,
    "temperature": 0.7
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | `"Hello!"` | The text prompt for generation. |
| `max_tokens` | integer | `256` | Maximum number of tokens to generate. |
| `temperature` | float | `0.7` | Sampling temperature (higher = more random). |

## Response format

```json
{
  "output": {
    "status": "success",
    "output": "What is the capital of France?\n\nThe capital of France is Paris."
  }
}
```

## Use a different model

Set the `MODEL_NAME` environment variable on your endpoint to any Hugging Face model ID that is available through model caching:

```
MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct
```

Make sure the model name in the endpoint's **Model** section matches the `MODEL_NAME` environment variable.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `microsoft/Phi-3-mini-4k-instruct` | Hugging Face model ID to load. |

## Troubleshooting

### Check if the model is cached

Add this to your handler or run it in the worker logs to verify the cache directory:

```python
import os

cache_root = "/runpod-volume/huggingface-cache/hub"

if os.path.exists(cache_root):
    print(f"Cache root exists: {cache_root}")
    for item in os.listdir(cache_root):
        print(f"  {item}")
else:
    print(f"Cache root does NOT exist: {cache_root}")
```

If the cache directory is empty or missing, make sure you added the model in the **Model** section when creating the endpoint.

### Common issues

| Issue | Solution |
|-------|----------|
| Model downloads instead of using cache | Add the model in the endpoint's **Model** section. The cache path must be `/runpod-volume/huggingface-cache`, not `/runpod/model-store/`. |
| "No space left on device" | Increase **Container Disk** to at least 20 GB when creating the endpoint. |
| Slow cold starts | Verify the model is cached (check logs for `[ModelStore] Using snapshot`). Set **Active Workers > 0** to keep workers warm. |
| `trust_remote_code` errors | Set `trust_remote_code=False` when the model is natively supported by `transformers`. Custom model code in cached snapshots can conflict with newer transformers versions. |
| Flash attention errors | Add `attn_implementation="eager"` to `from_pretrained()` if the base image does not include `flash-attn`. |
| PyTorch version mismatch | Use a base image with PyTorch >= 2.2. The `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` image works with current transformers versions. |

## Related

- [Deploy a cached model](https://docs.runpod.io/tutorials/serverless/model-caching-text) - Full tutorial for this worker.
- [Model caching](https://docs.runpod.io/serverless/endpoints/manage-endpoints#model-caching) - How model caching works on Runpod Serverless.
- [Runpod Serverless](https://docs.runpod.io/serverless/overview) - Serverless overview.
