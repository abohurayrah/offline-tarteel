# Whisper Fine-Tuning for Quranic Arabic

Fine-tune `openai/whisper-small` (244M parameters) on Quranic recitation data using LoRA, then export to ONNX for browser-based inference via transformers.js.

## Why

The current `tarteel-ai/whisper-tiny-ar-quran` achieves 79.2% accuracy overall but drops to **64.3% on non-professional reciters** (RetaSy source), often producing complete garbage transcripts. Whisper-small has 4x the capacity (12 encoder/decoder layers vs 4) and, combined with LoRA fine-tuning on diverse training data, should substantially improve accuracy on real-world recordings.

## Prerequisites

- **Python 3.10+** with pip
- **CUDA 11.8+** and an NVIDIA GPU (A100-80GB recommended, V100/A10 workable with reduced batch size)
- **HuggingFace account** with a write token (`HF_TOKEN`) for dataset access and model upload
- **~100 GB disk space** for datasets and checkpoints

```bash
pip install torch transformers[torch] peft datasets evaluate jiwer \
            accelerate soundfile librosa torchaudio tensorboard \
            huggingface-hub
```

For ONNX export:
```bash
pip install optimum[onnxruntime] onnxruntime onnx
```

For Modal cloud training:
```bash
pip install modal
modal token new
```

## Training Data

| Dataset | Hours | Reciters | Role |
|---------|-------|----------|------|
| `tarteel-ai/everyayah` | 829h train, 103h val | 36 professional | Primary training data |
| `RetaSy/quranic_audio_dataset` | varies | Non-professional | Diversity supplement |

The pipeline automatically loads and merges both datasets. If RetaSy is unavailable, training proceeds with EveryAyah only.

## Quick Start

### Option A: Run locally (A100 or similar)

```bash
export HF_TOKEN=hf_your_token_here

python scripts/finetune_whisper_quran.py
```

Training takes approximately **6-8 hours** on a single A100-80GB for 3 epochs.

### Option B: Run on Modal (recommended)

```bash
export HF_TOKEN=hf_your_token_here

# Create a Modal secret for HuggingFace access
modal secret create huggingface-secret HF_TOKEN=$HF_TOKEN

# Launch training
modal run scripts/finetune_whisper_modal.py

# Or train + export ONNX in one go
modal run scripts/finetune_whisper_modal.py --export
```

### Option C: Run on Lambda Labs / other cloud

SSH into a GPU instance, clone the repo, and run Option A. The script auto-detects CUDA and adjusts precision.

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_MODEL` | `openai/whisper-small` | HuggingFace model ID |
| `HUB_REPO` | `tarteel-ai/whisper-small-quran-lora` | Where to push the trained model |
| `EPOCHS` | `3` | Number of training epochs |
| `LR` | `1e-5` | Learning rate |
| `BATCH_SIZE` | `4` | Per-device batch size |
| `GRAD_ACCUM` | `8` | Gradient accumulation steps (effective batch = 4 x 8 = 32) |
| `LORA_R` | `32` | LoRA rank |
| `LORA_ALPHA` | `64` | LoRA alpha |
| `OUTPUT_DIR` | `./output/whisper-small-quran` | Local output directory |

Example with overrides:
```bash
EPOCHS=5 LR=5e-6 BATCH_SIZE=2 GRAD_ACCUM=16 python scripts/finetune_whisper_quran.py
```

## LoRA Details

We use LoRA (Low-Rank Adaptation) instead of full fine-tuning:
- **Target modules:** `q_proj`, `v_proj` in every attention layer
- **Rank 32, Alpha 64** (alpha/rank = 2.0 scaling factor)
- **Trainable parameters:** ~6M out of 244M total (~2.5%)
- The output projection (`proj_out`) is also kept trainable for vocabulary adaptation

After training, LoRA weights are merged back into the base model for export.

## Data Augmentation

Applied during training to improve robustness to real-world conditions:
- **Speed perturbation:** 0.9x and 1.1x playback speed (simulates natural tempo variation)
- **Noise injection:** Gaussian noise at 20dB SNR on 50% of samples (simulates ambient noise)
- **Short audio padding:** Clips under 1 second are padded with silence (prevents hallucination)

## Anti-Hallucination Measures

Whisper is prone to hallucination (generating repetitive or fabricated text) on short or noisy inputs. We mitigate this with:

1. **`no_repeat_ngram_size=3`** in generation config — prevents 3-gram repetition loops
2. **`condition_on_previous_text=False`** — prevents the model from building on its own mistakes
3. **Short audio padding** — Whisper hallucinates when audio is much shorter than its 30-second window
4. **forced_decoder_ids** — Forces Arabic language + transcribe task tokens at the start

## Export to ONNX

After training, export the merged model for browser use:

```bash
# From local model
python scripts/export_whisper_onnx.py

# From HuggingFace Hub
MODEL_PATH=your-org/whisper-small-quran python scripts/export_whisper_onnx.py

# On Modal (if you trained there)
modal run scripts/finetune_whisper_modal.py::export_onnx
```

This produces:
```
output/whisper-small-quran/onnx-export/
  onnx/
    encoder_model.onnx          # fp32 (~95 MB for whisper-small)
    decoder_model_merged.onnx   # q8  (~80 MB for whisper-small)
  config.json
  generation_config.json        # Has forced_decoder_ids + lang_to_id
  tokenizer.json
  tokenizer_config.json
  vocab.json
  merges.txt
  special_tokens_map.json
  added_tokens.json
  normalizer.json
  preprocessor_config.json
```

**Why fp32 encoder + q8 decoder?** The encoder uses Conv1D layers. WASM (the backend used by transformers.js) does not implement ConvInteger, so quantized convolutions crash. The decoder has only MatMul/Add/LayerNorm operations that quantize safely.

## Deploy to Web Frontend

Copy the exported model into the frontend's public assets:

```bash
# Back up current model
mv web/frontend/public/models/whisper-quran web/frontend/public/models/whisper-quran-tiny-backup

# Deploy new model
cp -r output/whisper-small-quran/onnx-export web/frontend/public/models/whisper-quran
```

No code changes are needed in the frontend. The `whisper-transcriber.ts` already loads from `/models/whisper-quran` with `dtype: { encoder_model: "fp32", decoder_model_merged: "q8" }`, which matches the export format exactly.

**Note:** The whisper-small model is larger than whisper-tiny (~175 MB total vs ~79 MB). Initial page load will take a few seconds longer. After the first load, the model is cached by the browser's Cache API.

## Expected Costs and Timing

| Platform | GPU | Training Time (3 epochs) | Cost |
|----------|-----|-------------------------|------|
| Modal | A100-80GB | ~6-8 hours | ~$15-25 |
| Lambda Labs | A100-80GB | ~6-8 hours | ~$10-15 |
| Lambda Labs | A10-24GB | ~18-24 hours | ~$10-15 |
| RunPod | A100-80GB | ~6-8 hours | ~$12-20 |

ONNX export takes ~30 minutes on CPU (no GPU needed).

## Output Files

After training completes:
```
output/whisper-small-quran/
  lora-adapter/           # LoRA weights only (~25 MB) — for further fine-tuning
  merged/                 # Full merged model (~500 MB) — for ONNX export or PyTorch inference
  checkpoints/            # Training checkpoints (auto-pruned to last 3)
  onnx-export/            # Browser-ready ONNX model (after export step)
  training_metrics.json   # Final loss and WER
```

## Troubleshooting

**CUDA out of memory:** Reduce `BATCH_SIZE` to 2 and increase `GRAD_ACCUM` to 16 to keep the same effective batch size. For GPUs under 40GB, also set `GRAD_ACCUM=32`.

**Dataset loading hangs:** The EveryAyah dataset is ~100 GB. First download takes 1-2 hours depending on bandwidth. Subsequent runs use the HuggingFace cache.

**transformers.js crashes after model swap:** Check that `generation_config.json` has `forced_decoder_ids` and `lang_to_id`. The export script ensures this, but manual copying can lose these fields.

**ONNX validation fails:** Ensure your `onnxruntime` version matches or is newer than the version used for export. Run `pip install --upgrade onnxruntime`.
