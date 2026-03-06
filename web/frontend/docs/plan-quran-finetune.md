# Plan: Fine-Tuning FastConformer on Quran Recitation Audio

## Problem Statement

Our current ASR model (`fastconformer_phoneme_q8.onnx`, ~126MB) is a **general Arabic ASR** model.
Its BPE vocabulary includes colloquial Arabic tokens ("عشان", "بتاع", "احنا") that are useless for
Quran recognition, and it has never been trained on the specific acoustic patterns of Quran recitation:

- **Tajweed rules**: elongation (madd), nasalization (ghunna), assimilation (idgham), pausing (waqf)
- **Recitation style**: slower, more deliberate pronunciation than conversational Arabic
- **Acoustic environment**: often recorded in studios with reverb/echo for spiritual effect
- **Vocabulary**: closed set of ~78,000 unique Arabic words (with diacritics) from the Quran

The model currently produces garbled output on ~2.5% of test cases (11 of 800 tests fail due to
bad transcription, not matching logic). Fine-tuning should directly address this.

## What We're Fine-Tuning

### Base Model
- **Architecture**: NVIDIA NeMo FastConformer-CTC
- **Size**: ~126MB quantized (int8), likely ~300-500MB full precision (fp32)
- **Input**: 80-mel spectrogram, 16kHz mono audio
- **Output**: 1025 BPE tokens (1024 Arabic subword tokens + 1 blank)
- **Preprocessing**: Pre-emphasis (0.97), dither (1e-5), Hann window (25ms), hop (10ms), per-feature normalization

### What Fine-Tuning Changes
- **Encoder weights**: Adjusted attention patterns to recognize Quran recitation acoustics
- **Decoder head**: Adjusted token probabilities to favor Quran-relevant BPE tokens
- **What stays the same**: Architecture, input shape, vocab size, preprocessing pipeline

## Training Data

### Primary Dataset: Buraaq/quran-md-ayahs (HuggingFace)
- ~150K+ audio-text pairs
- Multiple reciters: Alafasy, Husary, Minshawy, Abdul Basit, and others
- Per-ayah recordings with surah/ayah metadata
- Audio format: MP3 (various bitrates)

### Data Preparation Pipeline

```python
# 1. Download and convert to 16kHz mono WAV
# 2. Create NeMo-compatible manifest (JSONL format)
# 3. Split: 90% train, 5% val, 5% test

# NeMo manifest format (one JSON per line):
{"audio_filepath": "/data/quran/1_1_alafasy.wav", "text": "بسم الله الرحمن الرحيم", "duration": 3.2}
{"audio_filepath": "/data/quran/1_2_alafasy.wav", "text": "الحمد لله رب العالمين", "duration": 2.8}
```

### Text Processing Decisions

**Option A: Keep BPE tokens, Quran text only (recommended)**
- Use the existing 1025-token BPE vocabulary unchanged
- Training text = Quran verses in the same normalization the model was trained with
- Pro: No vocab change, no architecture change, smallest delta from base model
- Pro: Model learns to use existing tokens in Quran-specific patterns
- Con: Some BPE tokens will be wasted (colloquial words never appear in Quran)

**Option B: New BPE vocabulary (higher effort, higher ceiling)**
- Retrain BPE tokenizer on Quran text only
- Smaller, more efficient vocabulary (~500 tokens sufficient for Quran)
- Requires retraining the entire decoder head from scratch
- Much more training data needed, higher risk of overfitting

**Option C: Character-level / Phoneme-level (highest effort)**
- Replace BPE with per-character or per-phoneme output
- Perfect alignment with Quran trie (no BPE-to-char bridge needed)
- Requires significant architecture changes
- Longer sequences = slower inference

**Recommendation: Option A.** Keep existing BPE vocab. The model already knows how to produce
Arabic text — we just need it to produce *Quran-specific* Arabic text more reliably.

### Data Augmentation

Quran recitation has specific acoustic characteristics. Augment to build robustness:

```python
augmentations = {
    "speed_perturb": [0.9, 0.95, 1.0, 1.05, 1.1],  # Reciters vary speed
    "noise": {
        "types": ["white", "ambient_masjid"],
        "snr_db": [15, 20, 30, 40],
    },
    "reverb": {
        "room_sizes": ["small", "medium", "large"],  # Masjid reverb
    },
    "pitch_shift": [-1, 0, 1],  # Semi-tones, for voice variation
}
```

**Do NOT augment**: volume (already normalized by per-feature norm), time-stretch beyond ±10%
(tajweed elongation has specific rules).

### Multi-Reciter Strategy

Different reciters have dramatically different styles:
- **Husary**: Clear, pedagogical, slow — best for learning
- **Alafasy**: Modern, clear, moderate speed — most common in apps
- **Minshawy (Murattal)**: Traditional, some nasalization
- **Abdul Basit**: Very long madd, dramatic pauses

**Training split by reciter:**
- Include ALL reciters in training (generalization)
- Validation set: hold out one reciter entirely (e.g., Minshawy)
- This tests whether the model generalizes across recitation styles

## Training Configuration

### NeMo Training Script

```python
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from omegaconf import OmegaConf

# Load pretrained model
model = nemo_asr.models.EncDecCTCModel.from_pretrained(
    "nvidia/stt_ar_fastconformer_ctc_large"  # or whichever base model
)

# Override data config
model.cfg.train_ds.manifest_filepath = "/data/quran/train_manifest.jsonl"
model.cfg.train_ds.batch_size = 16
model.cfg.train_ds.num_workers = 4

model.cfg.validation_ds.manifest_filepath = "/data/quran/val_manifest.jsonl"
model.cfg.validation_ds.batch_size = 16

# Fine-tuning hyperparameters
model.cfg.optim.name = "adamw"
model.cfg.optim.lr = 1e-5          # Low LR for fine-tuning (not 1e-3!)
model.cfg.optim.weight_decay = 1e-4

# Cosine annealing with warm-up
model.cfg.optim.sched.name = "CosineAnnealing"
model.cfg.optim.sched.warmup_steps = 500
model.cfg.optim.sched.min_lr = 1e-7

# Freeze encoder for first N steps (optional — preserves acoustic features)
# model.encoder.freeze()  # Unfreeze after warmup if desired

trainer = pl.Trainer(
    max_epochs=20,
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed",
    accumulate_grad_batches=4,  # Effective batch size = 64
    val_check_interval=0.25,    # Validate 4x per epoch
    gradient_clip_val=1.0,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            monitor="val_wer",
            mode="min",
            save_top_k=3,
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_wer",
            patience=5,
            mode="min",
        ),
    ],
)

trainer.fit(model)
```

### Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 1e-5 | Fine-tuning, not training from scratch |
| Batch size | 16 (×4 accum = 64 effective) | Balance GPU memory vs gradient quality |
| Epochs | 20 (with early stopping) | Quran is a small, repetitive corpus |
| Warmup | 500 steps | Prevent catastrophic forgetting at start |
| Weight decay | 1e-4 | Light regularization |
| Gradient clipping | 1.0 | Stability |
| Precision | bf16-mixed | Speed + memory efficiency |

### Preventing Catastrophic Forgetting

Fine-tuning risks destroying the general Arabic knowledge that helps with:
- Different accents and speaking styles
- Background noise robustness
- General phonetic understanding

**Mitigation strategies:**
1. **Low learning rate** (1e-5 not 1e-3)
2. **Early stopping** on validation WER
3. **Optional: Encoder freezing** for first 2-3 epochs
4. **Optional: Mix in 10-20% general Arabic data** during training
5. **Checkpoint comparison**: Always compare against base model on our test set

## Export Pipeline: NeMo → ONNX → Quantized ONNX

### Step 1: Export to ONNX

```python
import nemo.collections.asr as nemo_asr

# Load best checkpoint
model = nemo_asr.models.EncDecCTCModel.restore_from(
    "checkpoints/best_model.nemo"
)

# Export to ONNX
model.export("fastconformer_quran_fp32.onnx")
```

### Step 2: Quantize to INT8

```python
import onnxruntime.quantization as quant

quant.quantize_dynamic(
    model_input="fastconformer_quran_fp32.onnx",
    model_output="fastconformer_quran_q8.onnx",
    weight_type=quant.QuantType.QInt8,
    per_channel=True,
    reduce_range=False,
)
```

### Step 3: Validate ONNX Output

```python
import onnxruntime as ort
import numpy as np

# Load both models
nemo_model = nemo_asr.models.EncDecCTCModel.restore_from("best_model.nemo")
onnx_session = ort.InferenceSession("fastconformer_quran_q8.onnx")

# Compare outputs on same input
mel_input = np.random.randn(1, 80, 200).astype(np.float32)
length_input = np.array([200], dtype=np.int64)

nemo_output = nemo_model.forward(...)  # Get logits
onnx_output = onnx_session.run(None, {
    "audio_signal": mel_input,
    "length": length_input
})[0]

# Check: max absolute difference should be < 0.1 for int8 quantization
print(f"Max diff: {np.max(np.abs(nemo_output - onnx_output)):.4f}")
```

### Step 4: Validate in Browser

The exported ONNX model is a **drop-in replacement** for the current model.
No code changes needed in the frontend — just swap the file:

```bash
# Replace the model file
cp fastconformer_quran_q8.onnx public/fastconformer_phoneme_q8.onnx

# The vocab stays the same (Option A), so no vocab.json changes needed
# Run the test suite to validate
npx tsx test/test-pipeline-full.ts --sample=200
```

## Compute Requirements

### GPU Training

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 1× RTX 3090 (24GB) | 1× A100 (40GB) |
| RAM | 32GB | 64GB |
| Disk | 50GB (dataset + checkpoints) | 100GB |
| Training time | ~4-8 hours (20 epochs) | ~2-4 hours |
| Cost (cloud) | ~$10-20 (Lambda, RunPod) | ~$20-40 |

### Cloud GPU Options

1. **Google Colab Pro** ($10/mo): T4/A100, good for prototyping
2. **Lambda Labs**: A100 at $1.10/hr, best value for serious training
3. **RunPod**: A100 at $1.04/hr serverless, flexible
4. **Vast.ai**: Cheapest, $0.50-1/hr for RTX 3090

### Without a GPU

If no GPU is available:
- **Google Colab Free**: T4 GPU, 12-hour sessions, enough for small experiments
- **Kaggle Notebooks**: Free P100, 30hrs/week
- **Mixed precision (bf16)**: Halves memory requirements

## Evaluation Strategy

### Metrics

1. **WER (Word Error Rate)**: Standard ASR metric, lower is better
2. **CER (Character Error Rate)**: More forgiving, better for Arabic
3. **Verse Match Accuracy**: Our end-to-end metric (transcribe → match → correct verse?)
4. **Equiv-Aware Accuracy**: Verse match accounting for identical-text verse groups

### Test Sets

| Test Set | Purpose | Size |
|----------|---------|------|
| Base validation | Standard WER/CER during training | 5% of dataset (~7.5K samples) |
| Held-out reciter | Generalization to unseen recitation style | ~30K samples (1 reciter) |
| Our test suite | End-to-end accuracy (the metric we actually care about) | 800 samples |
| Edge cases | Currently failing cases from test suite | 11 samples |

### A/B Comparison

Always compare fine-tuned model against base model:

```
                    Base Model    Fine-Tuned    Delta
Strict accuracy:    95.1%         ??.?%         +?.?%
Equiv-aware:        97.5%         ??.?%         +?.?%
Lenient(±2):        98.6%         ??.?%         +?.?%
Edge case fixes:    0/11          ?/11          +?
Model size (q8):    126MB         ~126MB        0
Inference time:     ~200ms        ~200ms        ~0
```

## Implementation Plan

### Phase 1: Data Preparation (1 day)

1. Download Buraaq/quran-md-ayahs dataset (we already have the download scripts)
2. Convert all audio to 16kHz mono WAV
3. Create NeMo JSONL manifest with proper text normalization
4. Split into train/val/test sets
5. Verify: spot-check 20 random audio-text pairs for alignment

**Output**: `data/quran/train_manifest.jsonl`, `val_manifest.jsonl`, `test_manifest.jsonl`

### Phase 2: Baseline Evaluation (0.5 days)

1. Set up NeMo environment (Docker or conda)
2. Load pretrained model, run on Quran test set
3. Compute WER/CER baseline
4. Identify which reciters/surahs have highest error rates
5. Save baseline numbers for comparison

**Output**: Baseline WER/CER numbers, per-reciter breakdown

### Phase 3: Fine-Tuning (1-2 days)

1. Set up training config (see script above)
2. Run initial training with encoder frozen (2-3 epochs)
3. Unfreeze encoder, continue training (remaining epochs)
4. Monitor val_wer, save best 3 checkpoints
5. If overfitting: increase dropout, add general Arabic data, reduce LR

**Output**: Best model checkpoint (`.nemo` format)

### Phase 4: Export and Validate (0.5 days)

1. Export best checkpoint to ONNX (fp32)
2. Quantize to INT8
3. Verify ONNX output matches NeMo output (within quantization tolerance)
4. Test in browser with our existing test suite
5. Compare accuracy against base model

**Output**: `fastconformer_quran_q8.onnx` (drop-in replacement)

### Phase 5: Iterate (1-2 days)

Based on Phase 4 results:
- If WER improved but some edge cases still fail → analyze failures, adjust training
- If overfitting → more augmentation, less epochs, lower LR
- If underfitting → more epochs, higher LR, unfreeze encoder earlier
- If perfect → celebrate, ship it

## Risks and Mitigations

### 1. Catastrophic Forgetting
**Risk**: Model loses general Arabic robustness, performs worse on noisy/partial audio.
**Mitigation**: Low LR, early stopping, mix in general data, always A/B test against base.

### 2. Overfitting to Specific Reciters
**Risk**: Model memorizes Alafasy's voice, fails on others.
**Mitigation**: Hold out one reciter for validation. Use speed/pitch augmentation.

### 3. Small Dataset
**Risk**: ~150K samples may not be enough for a large model.
**Mitigation**: Heavy augmentation. Fine-tune, don't train from scratch. Freeze encoder layers.

### 4. BPE Vocabulary Mismatch
**Risk**: Some Quran words require BPE combinations the model rarely produced before.
**Mitigation**: Option A minimizes this — we're not changing the vocab, just adjusting probabilities.
If specific BPE sequences are problematic, we'll see them in CER analysis.

### 5. Quantization Degradation
**Risk**: INT8 quantization loses the fine-tuning gains.
**Mitigation**: Compare fp32 and q8 ONNX outputs. If gap is too large, try fp16 quantization
(larger model but better accuracy). Our current q8 model works well, so this is low risk.

### 6. Model Size Regression
**Risk**: Fine-tuned model is larger or slower.
**Mitigation**: Same architecture = same size. INT8 quantization keeps it at ~126MB.

## Expected Impact

| Scenario | Expected Accuracy | Notes |
|----------|-------------------|-------|
| Current (general Arabic) | 97.5% equiv-aware | Base model, no fine-tuning |
| Light fine-tuning (5 epochs) | 98-98.5% | Quick win, low risk |
| Full fine-tuning (20 epochs) | 98.5-99.5% | Best expected outcome |
| Fine-tune + constrained decoding | 99-99.8% | Combined with Plan 1 |

## Appendix: NeMo Environment Setup

```bash
# Option 1: Docker (recommended)
docker pull nvcr.io/nvidia/nemo:24.01.speech

# Option 2: Conda
conda create -n nemo python=3.10
conda activate nemo
pip install nemo_toolkit[asr]
pip install pytorch-lightning onnxruntime

# Option 3: Google Colab
!pip install nemo_toolkit[asr] pytorch-lightning
```

## Appendix: Identifying the Exact Base Model

We need to identify which pretrained NeMo model was used to create our ONNX file.
Candidates from NVIDIA NGC:

1. `stt_ar_fastconformer_ctc_large` — Arabic FastConformer CTC (most likely)
2. `stt_ar_fastconformer_ctc_medium` — Smaller variant
3. Custom community model

**To verify:**
```python
# Check ONNX model metadata
import onnx
model = onnx.load("fastconformer_phoneme_q8.onnx")
for prop in model.metadata_props:
    print(f"{prop.key}: {prop.value}")

# Compare parameter count
# Our model (q8): 126MB → fp32 estimate: ~500MB
# NeMo large: ~120M params → ~480MB fp32
# This matches!
```

## Appendix: Alternative — LoRA Fine-Tuning

If full fine-tuning is too risky (catastrophic forgetting), consider LoRA:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                     # Low-rank dimension
    lora_alpha=32,            # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Only adapt attention
    lora_dropout=0.1,
    bias="none",
)

model = get_peft_model(base_model, lora_config)
# Only ~0.5% of parameters are trainable
# Much lower risk of catastrophic forgetting
# But may have lower ceiling than full fine-tuning
```

**Trade-off**: LoRA adapts fewer parameters (lower risk, lower ceiling) vs full fine-tuning
(higher risk, higher ceiling). For our use case with a closed vocabulary domain (Quran),
full fine-tuning is likely worth the risk since we have a clear evaluation metric.
