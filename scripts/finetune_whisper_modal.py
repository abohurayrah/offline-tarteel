"""
Modal wrapper for Whisper Quranic Arabic fine-tuning.

Runs the training script on cloud A100-80GB / H100 / H200 GPUs via Modal.
Handles dependency installation, GPU provisioning, and artifact download.

Usage:
    # Set your HuggingFace token for dataset access and model upload
    export HF_TOKEN=hf_...

    # Run training on Modal (1x A100-80GB, ~8-12 hours for 3 epochs)
    modal run scripts/finetune_whisper_modal.py

    # Multi-GPU (4x A100-80GB, ~2-3 hours, same total cost)
    NUM_GPUS=4 modal run scripts/finetune_whisper_modal.py

    # Use H100 instead (faster single-GPU, ~$3.95/hr vs $2.50/hr)
    GPU_TYPE=H100 modal run scripts/finetune_whisper_modal.py

    # Run with custom settings
    HUB_REPO=your-name/whisper-small-quran modal run scripts/finetune_whisper_modal.py

    # Download artifacts after training
    modal volume get whisper-quran-output /output ./local-output

Prerequisites:
    pip install modal
    modal token new    # authenticate with Modal
    modal secret create huggingface-secret HF_TOKEN=hf_...  # store HF token
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal configuration
# ---------------------------------------------------------------------------

app = modal.App("whisper-quran-finetune")

# Persistent volume for checkpoints and output — survives across runs so
# you can resume interrupted training or download artifacts later
output_volume = modal.Volume.from_name(
    "whisper-quran-output", create_if_missing=True
)

# GPU image with all training dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        # Core ML
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        # LoRA
        "peft>=0.7.0",
        # Data
        "datasets>=2.16.0",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        "torchaudio>=2.1.0",
        # Evaluation
        "evaluate>=0.4.0",
        "jiwer>=3.0.0",
        # Logging
        "tensorboard>=2.14.0",
        # HuggingFace Hub
        "huggingface-hub>=0.20.0",
    )
    # Mount the training script into the container
    .add_local_file(
        str(Path(__file__).parent / "finetune_whisper_quran.py"),
        remote_path="/app/finetune_whisper_quran.py",
    )
)


# ---------------------------------------------------------------------------
# GPU configuration
# ---------------------------------------------------------------------------
# Options (set NUM_GPUS env var, set GPU_TYPE to "H100" for H100):
#
# Modal pricing (as of 2026-03):
#   A100-80GB: ~$2.50/hr/GPU    H100: ~$3.95/hr/GPU    H200: ~$4.54/hr/GPU
#
# Estimated cost for 3 epochs of Whisper-small LoRA fine-tuning:
#   1x A100-80GB: ~8-12 hours, ~$20-30   (default, most reliable)
#   2x A100-80GB: ~4-6 hours,  ~$20-30   (same cost, 2x faster)
#   4x A100-80GB: ~2-3 hours,  ~$20-30   (same cost, 4x faster)
#   1x H100:      ~5-7 hours,  ~$20-28   (faster per GPU, good single-GPU option)
#   1x H200:      ~4-6 hours,  ~$18-27   (fastest single GPU)
#
# Multi-GPU uses accelerate's data-parallel training automatically.
# Total cost is similar because you rent more GPUs for less time.
# A100-80GB supports up to 8 GPUs per node on Modal.

NUM_GPUS: int = int(os.environ.get("NUM_GPUS", "1"))
GPU_TYPE: str = os.environ.get("GPU_TYPE", "A100-80GB")

def _gpu_spec() -> str:
    """Build the Modal GPU spec string.

    Modal's current API uses string specs: "A100-80GB", "H100:4", etc.
    The class-based API (modal.gpu.A100(...)) is deprecated.
    """
    gpu = GPU_TYPE
    if gpu not in ("A100-80GB", "A100-40GB", "H100", "H200", "L40S"):
        raise ValueError(f"Unsupported GPU_TYPE={gpu}. Use A100-80GB, H100, or H200.")
    if NUM_GPUS > 1:
        return f"{gpu}:{NUM_GPUS}"
    return gpu


# ---------------------------------------------------------------------------
# Training function — runs on A100/H100
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    gpu=_gpu_spec(),
    volumes={"/output": output_volume},
    timeout=12 * 60 * 60,  # 12 hours max
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def train():
    """Run the Whisper fine-tuning script on A100 GPU(s)."""
    import subprocess
    import sys

    # Forward environment variables to the training script
    env = os.environ.copy()
    env["OUTPUT_DIR"] = "/output/whisper-small-quran"

    # Use HF_TOKEN from Modal secret if available
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        env["HF_TOKEN"] = hf_token
        print("[Modal] HF_TOKEN found — model will be pushed to Hub after training")
    else:
        print("[Modal] No HF_TOKEN — model will only be saved to volume")

    # Forward any overrides from the caller
    for key in ["HUB_REPO", "EPOCHS", "LR", "BATCH_SIZE", "GRAD_ACCUM",
                "LORA_R", "LORA_ALPHA", "BASE_MODEL"]:
        val = os.environ.get(key)
        if val:
            env[key] = val

    # Multi-GPU: increase batch size proportionally
    if NUM_GPUS > 1:
        base_batch = int(env.get("BATCH_SIZE", "4"))
        env["BATCH_SIZE"] = str(base_batch)  # per-GPU batch stays same
        # Reduce grad accum since effective batch = per_gpu * num_gpus * grad_accum
        base_accum = int(env.get("GRAD_ACCUM", "8"))
        env["GRAD_ACCUM"] = str(max(1, base_accum // NUM_GPUS))

    print("=" * 60)
    print(f"Starting Whisper Quranic fine-tuning on Modal {NUM_GPUS}x {GPU_TYPE}")
    print("=" * 60)
    print(f"  GPUs:        {NUM_GPUS}x {GPU_TYPE}")
    print(f"  Output dir:  /output/whisper-small-quran")
    print(f"  HF Token:    {'set' if hf_token else 'not set'}")
    print(f"  Hub repo:    {env.get('HUB_REPO', 'tarteel-ai/whisper-small-quran-lora')}")
    if NUM_GPUS > 1:
        print(f"  Batch/GPU:   {env.get('BATCH_SIZE', '4')}")
        print(f"  Grad accum:  {env.get('GRAD_ACCUM', '?')}")
        print(f"  Effective:   {int(env.get('BATCH_SIZE','4')) * NUM_GPUS * int(env.get('GRAD_ACCUM','1'))}")
    print()

    # For multi-GPU: use accelerate to launch distributed training.
    # Modal provisions all GPUs on a single node, so we use single-machine
    # multi-GPU mode. accelerate handles DistributedDataParallel automatically.
    if NUM_GPUS > 1:
        cmd = [
            sys.executable, "-m", "accelerate", "launch",
            "--multi_gpu",
            "--num_processes", str(NUM_GPUS),
            "--num_machines", "1",
            "--mixed_precision", "fp16",
            "/app/finetune_whisper_quran.py",
        ]
    else:
        cmd = [sys.executable, "/app/finetune_whisper_quran.py"]

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Training script failed with exit code {result.returncode}")

    # Commit the volume so artifacts persist
    output_volume.commit()

    print()
    print("=" * 60)
    print("Training complete! Artifacts saved to Modal volume.")
    print("=" * 60)
    print()
    print("To download the trained model:")
    print("  modal volume get whisper-quran-output /whisper-small-quran/merged ./output/merged")
    print()
    print("To download the LoRA adapter only:")
    print("  modal volume get whisper-quran-output /whisper-small-quran/lora-adapter ./output/lora-adapter")


# ---------------------------------------------------------------------------
# ONNX export — can run on a cheaper GPU or CPU
# ---------------------------------------------------------------------------

export_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "optimum[onnxruntime]>=1.16.0",
        "onnxruntime>=1.16.0",
        "onnx>=1.15.0",
    )
    .add_local_file(
        str(Path(__file__).parent / "export_whisper_onnx.py"),
        remote_path="/app/export_whisper_onnx.py",
    )
)


@app.function(
    image=export_image,
    volumes={"/output": output_volume},
    timeout=60 * 60,  # 1 hour
    memory=32768,  # 32 GB RAM — ONNX export is memory-hungry
    cpu=8,
)
def export_onnx():
    """Export the trained model to ONNX on Modal (no GPU needed)."""
    import subprocess
    import sys

    env = os.environ.copy()
    env["MODEL_PATH"] = "/output/whisper-small-quran/merged"
    env["OUTPUT_DIR"] = "/output/whisper-small-quran/onnx-export"

    print("=" * 60)
    print("Exporting trained Whisper model to ONNX")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "/app/export_whisper_onnx.py"],
        env=env,
        capture_output=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"ONNX export failed with exit code {result.returncode}")

    output_volume.commit()

    print()
    print("ONNX export complete! Download with:")
    print("  modal volume get whisper-quran-output /whisper-small-quran/onnx-export ./onnx-export")


# ---------------------------------------------------------------------------
# Download helper — copy artifacts from volume to local machine
# ---------------------------------------------------------------------------

@app.function(
    volumes={"/output": output_volume},
    timeout=300,
)
def list_artifacts():
    """List all training artifacts in the Modal volume."""
    import subprocess
    result = subprocess.run(
        ["find", "/output", "-type", "f", "-name", "*.json", "-o",
         "-name", "*.onnx", "-o", "-name", "*.safetensors", "-o",
         "-name", "*.bin"],
        capture_output=True, text=True,
    )
    print("Artifacts in volume:")
    print(result.stdout)
    return result.stdout


# ---------------------------------------------------------------------------
# Entrypoint — orchestrates train + optional export
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    export: bool = False,
    list_files: bool = False,
):
    """Run Whisper Quranic fine-tuning on Modal.

    Args:
        export: Also export to ONNX after training (adds ~30 min).
        list_files: Just list artifacts in the volume, don't train.
    """
    if list_files:
        list_artifacts.remote()
        return

    print(f"Launching training on Modal {NUM_GPUS}x {GPU_TYPE}...")
    if NUM_GPUS == 1:
        print("Estimated: 8-12 hours for 3 epochs. ~$20-30.")
        print("Tip: NUM_GPUS=4 modal run ... for 2-3 hours at same cost.")
    else:
        print(f"Estimated: {max(2, 10 // NUM_GPUS)}-{max(3, 12 // NUM_GPUS)} hours for 3 epochs. ~$20-30.")
    print()

    # Run training
    train.remote()

    # Optionally export to ONNX
    if export:
        print()
        print("Training done. Starting ONNX export...")
        export_onnx.remote()

    print()
    print("All done! Download your model with:")
    print("  modal volume get whisper-quran-output /whisper-small-quran/merged ./output/merged")
    if export:
        print("  modal volume get whisper-quran-output /whisper-small-quran/onnx-export ./output/onnx-export")
