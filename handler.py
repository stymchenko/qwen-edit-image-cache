import os
import sys
import runpod
import torch
#import torch._dynamo
import base64
import io
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

MODEL_ID = os.environ.get("MODEL_NAME", "Qwen/Qwen-Image-Edit-2511")
MAX_IMAGE_SIZE = int(os.environ.get("MAX_IMAGE_SIZE", '768'))
NUM_INFERENCE_STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", '40'))
GUIDANCE_SCALE = float(os.environ.get("GUIDANCE_SCALE", '1.0'))
TRUE_CFG_SCALE = float(os.environ.get("TRUE_CFG_SCALE", '4.0'))
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"

# Force offline mode to use only cached models
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print(f"[Startup] Python version: {sys.version}", flush=True)
print(f"[Startup] HF_CACHE_ROOT exists: {os.path.exists(HF_CACHE_ROOT)}", flush=True)
print(f"[Startup] Contents: {os.listdir(HF_CACHE_ROOT) if os.path.exists(HF_CACHE_ROOT) else 'NOT FOUND'}", flush=True)


def resolve_snapshot_path(model_id: str) -> str:
    """
    Resolve the local snapshot path for a cached model.

    Args:
        model_id: The model name from Hugging Face (e.g., 'Qwen/Qwen-Image-Edit-2511')

    Returns:
        The full path to the cached model snapshot
    """
    if "/" not in model_id:
        raise ValueError(f"MODEL_ID '{model_id}' is not in 'org/name' format")

    org, name = model_id.split("/", 1)
    model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")

    print(f"[ModelStore] MODEL_ID: {model_id}", flush=True)
    print(f"[ModelStore] Model root: {model_root}", flush=True)

    # Try to read the snapshot hash from refs/main
    if os.path.isfile(refs_main):
        with open(refs_main, "r") as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            print(f"[ModelStore] Using snapshot from refs/main: {candidate}", flush=True)
            return candidate

    # Fall back to first available snapshot
    if not os.path.isdir(snapshots_dir):
        raise RuntimeError(f"[ModelStore] snapshots directory not found: {snapshots_dir}")

    versions = [
        d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))
    ]

    if not versions:
        raise RuntimeError(f"[ModelStore] No snapshot subdirectories found under {snapshots_dir}")

    versions.sort()
    chosen = os.path.join(snapshots_dir, versions[0])
    print(f"[ModelStore] Using first available snapshot: {chosen}", flush=True)
    return chosen


# Resolve and load the model at startup
LOCAL_MODEL_PATH = resolve_snapshot_path(MODEL_ID)
print(f"[ModelStore] Resolved local model path: {LOCAL_MODEL_PATH}", flush=True)

model = QwenImageEditPlusPipeline.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    # device_map="cuda"
    # Це прибере помилку зі схемами, оскільки Flash Attention 2
    # має власний шлях ініціалізації
    attn_implementation="flash_attention_2"
).to("cuda")

print("[ModelStore] Model loaded from local snapshot", flush=True)

# torch._dynamo.config.suppress_errors = True
# model.transformer = torch.compile(model.transformer, mode="reduce-overhead")

print("[ModelStore] Model compiled", flush=True)

def handler(job):
    """
    Handler function that processes each inference request.

    Args:
        job: Runpod job object containing input data

    Returns:
        Dictionary with generated text or error information
    """
    job_input = job["input"]
    prompt = job_input.get("prompt")
    image_base64 = job_input.get("image")  # Очікуємо base64 рядок

    if not prompt or not image_base64:
        return {"error": "Необхідно надати prompt та image (base64)"}

    try:
        # Декодування зображення
        image_data = base64.b64decode(image_base64)
        init_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Ресайз до максимум MAX_IMAGE_SIZE по більшій стороні зі збереженням пропорцій
        w, h = init_image.size
        if max(w, h) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(w, h)
            init_image = init_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # Генерація
        output = model(
            prompt=prompt,
            negative_prompt="",
            image=init_image,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            true_cfg_scale=TRUE_CFG_SCALE,
            num_images_per_prompt=1
        ).images[0]

        # Кодування результату в base64
        buffered = io.BytesIO()
        output.save(buffered, format="JPEG", quality=90)
        output_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image": output_base64, "format": "jpeg"}

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
