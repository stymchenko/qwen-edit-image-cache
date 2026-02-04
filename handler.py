import os
import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_ID = os.environ.get("MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"

# Force offline mode to use only cached models
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def resolve_snapshot_path(model_id: str) -> str:
    """
    Resolve the local snapshot path for a cached model.

    Args:
        model_id: The model name from Hugging Face (e.g., 'microsoft/Phi-3-mini-4k-instruct')

    Returns:
        The full path to the cached model snapshot
    """
    if "/" not in model_id:
        raise ValueError(f"MODEL_ID '{model_id}' is not in 'org/name' format")

    org, name = model_id.split("/", 1)
    model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")

    print(f"[ModelStore] MODEL_ID: {model_id}")
    print(f"[ModelStore] Model root: {model_root}")

    # Try to read the snapshot hash from refs/main
    if os.path.isfile(refs_main):
        with open(refs_main, "r") as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            print(f"[ModelStore] Using snapshot from refs/main: {candidate}")
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
    print(f"[ModelStore] Using first available snapshot: {chosen}")
    return chosen


# Resolve and load the model at startup
LOCAL_MODEL_PATH = resolve_snapshot_path(MODEL_ID)
print(f"[ModelStore] Resolved local model path: {LOCAL_MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=False,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=False,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True,
    attn_implementation="eager",
)

text_gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

print("[ModelStore] Model loaded from local snapshot")


def handler(job):
    """
    Handler function that processes each inference request.

    Args:
        job: Runpod job object containing input data

    Returns:
        Dictionary with generated text or error information
    """
    job_input = job.get("input", {}) or {}
    prompt = job_input.get("prompt", "Hello!")
    max_tokens = int(job_input.get("max_tokens", 256))
    temperature = float(job_input.get("temperature", 0.7))

    print(f"[Handler] Prompt: {prompt[:80]!r}")
    print(f"[Handler] max_tokens={max_tokens}, temperature={temperature}")

    try:
        outputs = text_gen(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
        )
        generated = outputs[0]["generated_text"]
        print(f"[Handler] Generated length: {len(generated)} chars")

        return {
            "status": "success",
            "output": generated,
        }

    except Exception as e:
        print(f"[Handler] Error during generation: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


runpod.serverless.start({"handler": handler})
