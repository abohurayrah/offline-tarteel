"""
Fine-tune openai/whisper-small on Quranic Arabic recitation data.

Produces a LoRA-adapted model optimized for diverse reciters (professional and
non-professional). Designed to run on a single A100-80GB GPU but gracefully
falls back to smaller GPUs via gradient checkpointing + reduced batch size.

Usage:
    python scripts/finetune_whisper_quran.py

    # Override defaults via environment variables:
    HF_TOKEN=hf_...  HUB_REPO=your-org/whisper-small-quran  \
    EPOCHS=3  LR=1e-5  BATCH_SIZE=4  GRAD_ACCUM=8  \
    python scripts/finetune_whisper_quran.py

Requirements:
    pip install transformers[torch] peft datasets evaluate jiwer \
               accelerate soundfile librosa huggingface_hub
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download, login as hf_login
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Model and data
BASE_MODEL: str = os.getenv("BASE_MODEL", "openai/whisper-small")
HUB_REPO: str = os.getenv("HUB_REPO", "tarteel-ai/whisper-small-quran-lora")
EVERYAYAH_DATASET: str = "tarteel-ai/everyayah"
RETASY_DATASET: str = "RetaSy/quranic_audio_dataset"
TARTEEL_DATASET: str = "ashraf-ali/quran-data"  # 25k Tarteel.io user recordings (18k labeled)

# LoRA hyperparameters
LORA_R: int = int(os.getenv("LORA_R", "32"))
LORA_ALPHA: int = int(os.getenv("LORA_ALPHA", "64"))
LORA_DROPOUT: float = float(os.getenv("LORA_DROPOUT", "0.05"))
LORA_TARGET_MODULES: list[str] = ["q_proj", "v_proj"]

# Training hyperparameters
NUM_EPOCHS: int = int(os.getenv("EPOCHS", "3"))
LEARNING_RATE: float = float(os.getenv("LR", "1e-5"))
PER_DEVICE_BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "4"))
GRADIENT_ACCUMULATION_STEPS: int = int(os.getenv("GRAD_ACCUM", "8"))
WARMUP_RATIO: float = 0.05
WEIGHT_DECAY: float = 0.01
FP16: bool = torch.cuda.is_available()

# Data augmentation
SPEED_PERTURBATION: bool = True
SPEED_FACTORS: list[float] = [0.9, 1.0, 1.1]
NOISE_INJECTION: bool = True
NOISE_SNR_DB: float = 20.0  # Signal-to-noise ratio in dB

# HuggingFace authentication (required for large dataset downloads to avoid rate limits)
HF_TOKEN: str | None = os.getenv("HF_TOKEN")

# Paths
OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "./output/whisper-small-quran"))
CHECKPOINT_DIR: Path = OUTPUT_DIR / "checkpoints"

# Audio constants
SAMPLING_RATE: int = 16_000
MAX_AUDIO_LENGTH_S: float = 30.0  # Whisper's max input length


def _authenticate_hf() -> None:
    """Authenticate with HuggingFace Hub to avoid rate limits on large downloads.

    The EveryAyah dataset alone is ~100GB. Without auth, HF will throttle
    downloads to ~5MB/s and disconnect after ~10 minutes. With a free HF
    token, you get full speed and no disconnects.

    Set HF_TOKEN env var or run `huggingface-cli login` before training.
    """
    if HF_TOKEN:
        hf_login(token=HF_TOKEN)
        logger.info("Authenticated with HuggingFace Hub via HF_TOKEN")
    elif Path.home().joinpath(".cache/huggingface/token").exists():
        logger.info("Using cached HuggingFace token from `huggingface-cli login`")
    else:
        logger.warning(
            "No HF_TOKEN set and no cached token found. "
            "Large dataset downloads WILL be rate-limited. "
            "Run `huggingface-cli login` or set HF_TOKEN=hf_... to fix this."
        )


# ---------------------------------------------------------------------------
# Data augmentation utilities
# ---------------------------------------------------------------------------

def speed_perturb(audio: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Change playback speed without changing pitch (via resampling)."""
    if abs(factor - 1.0) < 1e-6:
        return audio
    # Simple resampling: stretch/compress the time axis
    indices = np.round(np.arange(0, len(audio), factor)).astype(int)
    indices = indices[indices < len(audio)]
    return audio[indices]


def add_noise(audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Add Gaussian noise at a specified SNR level."""
    rms_signal = np.sqrt(np.mean(audio**2))
    if rms_signal < 1e-10:
        return audio
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.normal(0, rms_noise, audio.shape)
    return (audio + noise).astype(audio.dtype)


def pad_short_audio(audio: np.ndarray, sr: int, min_duration_s: float = 1.0) -> np.ndarray:
    """Pad audio shorter than min_duration_s with silence.

    Short audio clips cause Whisper to hallucinate because the model tries to
    fill the 30-second context window. Padding to at least 1 second mitigates
    this.
    """
    min_samples = int(min_duration_s * sr)
    if len(audio) >= min_samples:
        return audio
    padding = np.zeros(min_samples - len(audio), dtype=audio.dtype)
    return np.concatenate([audio, padding])


# ---------------------------------------------------------------------------
# Tarteel CSV + direct WAV download loader
# ---------------------------------------------------------------------------

def _load_tarteel_csv(
    processor: WhisperProcessor,
    hf_token: str | None,
    augment: bool = False,
) -> dict[str, list]:
    """Load Tarteel dataset from CSV metadata + direct WAV file downloads.

    The HuggingFace ``datasets`` library requires ``torchcodec`` for audio
    decoding, which fails in many environments. The Tarteel dataset also
    takes 4+ hours to download through the ``datasets`` streaming API.

    This loader bypasses both issues by:
    1. Downloading the CSV metadata via ``hf_hub_download`` (instant).
    2. Downloading individual WAV files on demand (cached after first fetch).
    3. Decoding audio with ``soundfile`` (pure C library, no torch needed).

    Returns a dict with ``input_features`` and ``labels`` lists ready for
    training.
    """
    repo = TARTEEL_DATASET
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    # ------------------------------------------------------------------
    # 1. Download and parse CSV metadata for both user and qari recordings
    # ------------------------------------------------------------------
    csv_files = ["users_ayahs.csv", "qari_short_ayahs.csv"]
    samples: list[dict[str, Any]] = []

    for csv_name in csv_files:
        try:
            csv_path = hf_hub_download(
                repo, csv_name, repo_type="dataset", token=hf_token,
            )
        except Exception as e:
            logger.warning("Could not download %s from %s: %s", csv_name, repo, e)
            continue

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                transcript = (row.get("transcript") or "").strip()
                file_name = (row.get("file_name") or "").strip()
                if not transcript or not file_name:
                    continue
                duration = float(row.get("duration_in_seconds") or 0)
                samples.append({
                    "file_path": file_name,
                    "text": transcript,
                    "duration": duration,
                })

        logger.info("Parsed %s: running total %d samples", csv_name, len(samples))

    if not samples:
        logger.warning("No Tarteel samples found in CSV files")
        return {"input_features": [], "labels": []}

    # ------------------------------------------------------------------
    # 2. Duration filter — skip very short (<0.5 s) or very long (>30 s)
    # ------------------------------------------------------------------
    before = len(samples)
    samples = [s for s in samples if 0.5 <= s["duration"] <= 30.0]
    logger.info(
        "Tarteel duration filter: %d -> %d samples (dropped %d)",
        before, len(samples), before - len(samples),
    )

    # ------------------------------------------------------------------
    # 3. Download WAVs and extract features
    # ------------------------------------------------------------------
    processed: dict[str, list] = {"input_features": [], "labels": []}
    errors = 0

    for i, sample in enumerate(samples):
        try:
            # Download WAV (hf_hub_download caches automatically)
            wav_path = hf_hub_download(
                repo, sample["file_path"],
                repo_type="dataset", token=hf_token,
            )
            audio_array, sr = sf.read(wav_path, dtype="float32")

            # Stereo -> mono
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)

            # Resample to 16 kHz if needed
            if sr != SAMPLING_RATE:
                import librosa
                audio_array = librosa.resample(
                    audio_array, orig_sr=sr, target_sr=SAMPLING_RATE,
                )

            # Truncate to Whisper's max input length
            max_samples = int(MAX_AUDIO_LENGTH_S * SAMPLING_RATE)
            if len(audio_array) > max_samples:
                audio_array = audio_array[:max_samples]

            # Anti-hallucination: pad very short clips
            audio_array = pad_short_audio(audio_array, SAMPLING_RATE, min_duration_s=1.0)

            # Data augmentation (training only)
            if augment:
                if SPEED_PERTURBATION:
                    factor = random.choice(SPEED_FACTORS)
                    audio_array = speed_perturb(audio_array, SAMPLING_RATE, factor)
                if NOISE_INJECTION and random.random() < 0.5:
                    audio_array = add_noise(audio_array, snr_db=NOISE_SNR_DB)

            # Extract mel-spectrogram features
            inputs = feature_extractor(
                audio_array, sampling_rate=SAMPLING_RATE, return_tensors="np",
            )
            input_features = inputs.input_features[0]

            # Tokenize transcript
            labels = tokenizer(sample["text"]).input_ids

            processed["input_features"].append(input_features)
            processed["labels"].append(labels)

        except Exception as e:
            errors += 1
            if errors <= 10:
                logger.warning("  Tarteel error on %s: %s", sample["file_path"], e)

        if (i + 1) % 2000 == 0:
            logger.info(
                "  Tarteel progress: %d/%d processed (%d errors so far)",
                i + 1, len(samples), errors,
            )

    logger.info(
        "Tarteel: %d samples processed, %d errors",
        len(processed["input_features"]), errors,
    )
    return processed


# ---------------------------------------------------------------------------
# Combined PyTorch Dataset (HF datasets + Tarteel dict data)
# ---------------------------------------------------------------------------

class CombinedQuranDataset(torch.utils.data.Dataset):
    """Combines a HuggingFace ``Dataset`` (EveryAyah / RetaSy) with the
    Tarteel dict-of-lists produced by ``_load_tarteel_csv``.

    Both sources yield ``{"input_features": ..., "labels": ...}`` items.
    """

    def __init__(
        self,
        hf_dataset,
        tarteel_data: dict[str, list] | None = None,
    ):
        self.hf_dataset = hf_dataset
        self.tarteel = tarteel_data or {"input_features": [], "labels": []}
        self._hf_len = len(hf_dataset) if hf_dataset is not None else 0
        self._tt_len = len(self.tarteel["input_features"])

    def __len__(self) -> int:
        return self._hf_len + self._tt_len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < self._hf_len:
            return self.hf_dataset[idx]
        tidx = idx - self._hf_len
        return {
            "input_features": self.tarteel["input_features"][tidx],
            "labels": self.tarteel["labels"][tidx],
        }


# ---------------------------------------------------------------------------
# Dataset loading and preprocessing
# ---------------------------------------------------------------------------

def load_and_prepare_datasets(
    processor: WhisperProcessor,
) -> tuple[CombinedQuranDataset, CombinedQuranDataset | None]:
    """Load, merge, and preprocess training datasets.

    Data sources (in priority order):
    1. tarteel-ai/everyayah (829h, 36 professional reciters) -- backbone accuracy
       Loaded via HF ``datasets`` (parquet-based, no audio decoder needed).
    2. ashraf-ali/quran-data (25k Tarteel.io user recordings) -- diversity/robustness
       Loaded via CSV + direct WAV download (bypasses broken torchcodec).
    3. RetaSy/quranic_audio_dataset (7k non-Arabic speakers) -- accent generalization
       Loaded via HF ``datasets``; skipped if no text column is found.

    Returns ``(train_dataset, val_dataset)`` where each is a
    :class:`CombinedQuranDataset` (or ``None`` for val).
    """

    # ------------------------------------------------------------------
    # Helper: detect audio / text column names
    # ------------------------------------------------------------------
    def detect_columns(dataset):
        """Detect audio and text column names from the dataset."""
        sample_cols = (
            list(dataset.column_names.values())[0]
            if isinstance(dataset.column_names, dict)
            else dataset.column_names
        )
        audio_col = next(
            (c for c in ["audio", "audio_path", "path", "file"] if c in sample_cols),
            None,
        )
        text_col = next(
            (c for c in ["text", "transcription", "sentence", "ayah_text", "label"]
             if c in sample_cols),
            None,
        )
        return audio_col, text_col

    # ------------------------------------------------------------------
    # Helper: preprocessing function factory for HF datasets
    # ------------------------------------------------------------------
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    def make_preprocess_fn(audio_col: str, text_col: str, augment: bool = False):
        """Create a preprocessing function for ``Dataset.map()``."""

        def preprocess(batch):
            audios = batch[audio_col]
            texts = batch[text_col]

            input_features_list = []
            labels_list = []

            for audio_item, text in zip(audios, texts):
                # Handle different audio formats from datasets library
                if isinstance(audio_item, dict):
                    audio_array = np.array(audio_item["array"], dtype=np.float32)
                    sr = audio_item["sampling_rate"]
                else:
                    audio_array = np.array(audio_item, dtype=np.float32)
                    sr = SAMPLING_RATE

                # Resample if needed
                if sr != SAMPLING_RATE:
                    import librosa
                    audio_array = librosa.resample(
                        audio_array, orig_sr=sr, target_sr=SAMPLING_RATE,
                    )

                # Truncate to max length
                max_samples = int(MAX_AUDIO_LENGTH_S * SAMPLING_RATE)
                if len(audio_array) > max_samples:
                    audio_array = audio_array[:max_samples]

                # Anti-hallucination: pad short audio to at least 1 second
                audio_array = pad_short_audio(audio_array, SAMPLING_RATE, min_duration_s=1.0)

                # Data augmentation (training only)
                if augment:
                    if SPEED_PERTURBATION:
                        factor = random.choice(SPEED_FACTORS)
                        audio_array = speed_perturb(audio_array, SAMPLING_RATE, factor)
                    if NOISE_INJECTION and random.random() < 0.5:
                        audio_array = add_noise(audio_array, snr_db=NOISE_SNR_DB)

                # Extract mel features
                features = feature_extractor(
                    audio_array,
                    sampling_rate=SAMPLING_RATE,
                    return_tensors="np",
                )
                input_features_list.append(features.input_features[0])

                # Tokenize text
                text = str(text).strip()
                label_ids = tokenizer(text).input_ids
                labels_list.append(label_ids)

            return {
                "input_features": input_features_list,
                "labels": labels_list,
            }

        return preprocess

    # ==================================================================
    # 1. EveryAyah — via HF datasets (parquet, no audio decoder issues)
    # ==================================================================
    logger.info("Loading primary dataset: %s", EVERYAYAH_DATASET)
    try:
        everyayah = load_dataset(EVERYAYAH_DATASET, token=HF_TOKEN)
        logger.info(
            "EveryAyah loaded: %d train, %d validation",
            len(everyayah.get("train", [])),
            len(everyayah.get("validation", [])),
        )
    except Exception as e:
        logger.error("Failed to load EveryAyah dataset: %s", e)
        logger.info("Attempting to load with trust_remote_code...")
        everyayah = load_dataset(
            EVERYAYAH_DATASET, streaming=False, trust_remote_code=True,
        )

    ea_audio_col, ea_text_col = detect_columns(everyayah)
    logger.info("EveryAyah columns: audio=%s, text=%s", ea_audio_col, ea_text_col)

    # Ensure audio is at 16 kHz
    if ea_audio_col:
        for split in everyayah:
            everyayah[split] = everyayah[split].cast_column(
                ea_audio_col, Audio(sampling_rate=SAMPLING_RATE),
            )

    # Collect HF-based train/val lists: (name, dataset, audio_col, text_col)
    hf_train_datasets: list[tuple[str, Any, str, str]] = []
    hf_val_datasets: list[tuple[str, Any, str, str]] = []

    if "train" in everyayah:
        hf_train_datasets.append(("everyayah", everyayah["train"], ea_audio_col, ea_text_col))
    if "validation" in everyayah:
        hf_val_datasets.append(("everyayah", everyayah["validation"], ea_audio_col, ea_text_col))
    elif "test" in everyayah:
        hf_val_datasets.append(("everyayah", everyayah["test"], ea_audio_col, ea_text_col))

    # ==================================================================
    # 2. RetaSy — via HF datasets; skip if no text column found
    # ==================================================================
    try:
        logger.info("Loading secondary dataset: %s", RETASY_DATASET)
        retasy = load_dataset(RETASY_DATASET, token=HF_TOKEN)
        logger.info("RetaSy loaded: %s", {k: len(v) for k, v in retasy.items()})

        rt_audio_col, rt_text_col = detect_columns(retasy)
        logger.info("RetaSy columns: audio=%s, text=%s", rt_audio_col, rt_text_col)

        if rt_audio_col and rt_text_col:
            for split in retasy:
                retasy[split] = retasy[split].cast_column(
                    rt_audio_col, Audio(sampling_rate=SAMPLING_RATE),
                )
            if "train" in retasy:
                hf_train_datasets.append(("retasy", retasy["train"], rt_audio_col, rt_text_col))
            if "validation" in retasy:
                hf_val_datasets.append(("retasy", retasy["validation"], rt_audio_col, rt_text_col))
            elif "test" in retasy:
                hf_val_datasets.append(("retasy", retasy["test"], rt_audio_col, rt_text_col))
        else:
            logger.warning(
                "RetaSy: skipping — no text column found (audio=%s, text=%s)",
                rt_audio_col, rt_text_col,
            )
    except Exception as e:
        logger.warning("Could not load RetaSy dataset (non-fatal): %s", e)

    # ==================================================================
    # 3. Tarteel — via CSV + direct WAV download (bypasses torchcodec)
    # ==================================================================
    logger.info("Loading Tarteel dataset via CSV: %s", TARTEEL_DATASET)
    tarteel_train_data = _load_tarteel_csv(
        processor, hf_token=HF_TOKEN, augment=True,
    )

    # Build a small validation slice from Tarteel (last 10 %)
    tarteel_val_data: dict[str, list] = {"input_features": [], "labels": []}
    if tarteel_train_data["input_features"]:
        n_total = len(tarteel_train_data["input_features"])
        n_val = max(1, int(n_total * 0.1))
        n_train = n_total - n_val
        tarteel_val_data = {
            "input_features": tarteel_train_data["input_features"][n_train:],
            "labels": tarteel_train_data["labels"][n_train:],
        }
        tarteel_train_data = {
            "input_features": tarteel_train_data["input_features"][:n_train],
            "labels": tarteel_train_data["labels"][:n_train],
        }
        logger.info(
            "Tarteel split: %d train, %d val", n_train, n_val,
        )

    # ==================================================================
    # 4. Preprocess HF-based datasets via .map()
    # ==================================================================
    processed_hf_train = []
    for name, ds, audio_col, text_col in hf_train_datasets:
        logger.info("Preprocessing %s train split (%d samples)...", name, len(ds))
        processed = ds.map(
            make_preprocess_fn(audio_col, text_col, augment=True),
            batched=True,
            batch_size=32,
            remove_columns=ds.column_names,
            num_proc=4,
            desc=f"Preprocessing {name} train",
        )
        processed_hf_train.append(processed)

    processed_hf_val = []
    for name, ds, audio_col, text_col in hf_val_datasets:
        logger.info("Preprocessing %s validation split (%d samples)...", name, len(ds))
        processed = ds.map(
            make_preprocess_fn(audio_col, text_col, augment=False),
            batched=True,
            batch_size=32,
            remove_columns=ds.column_names,
            num_proc=4,
            desc=f"Preprocessing {name} val",
        )
        processed_hf_val.append(processed)

    # Concatenate all HF-processed datasets
    hf_train = concatenate_datasets(processed_hf_train) if processed_hf_train else None
    hf_val = concatenate_datasets(processed_hf_val) if processed_hf_val else None

    # ==================================================================
    # 5. Combine HF datasets + Tarteel into CombinedQuranDataset
    # ==================================================================
    train_dataset = CombinedQuranDataset(hf_train, tarteel_train_data)
    val_dataset: CombinedQuranDataset | None = None
    if hf_val is not None or tarteel_val_data["input_features"]:
        val_dataset = CombinedQuranDataset(hf_val, tarteel_val_data)

    if len(train_dataset) == 0:
        raise RuntimeError("No training data loaded. Check dataset availability.")

    logger.info(
        "Final dataset sizes: train=%d (HF=%d + Tarteel=%d), val=%s (HF=%s + Tarteel=%d)",
        len(train_dataset),
        train_dataset._hf_len,
        train_dataset._tt_len,
        len(val_dataset) if val_dataset else "None",
        val_dataset._hf_len if val_dataset else 0,
        val_dataset._tt_len if val_dataset else 0,
    )

    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class WhisperDataCollator:
    """Collate Whisper features and labels into padded batches.

    Features are already fixed-size (80 x 3000 mel frames), but labels need
    padding. We pad labels with -100 so the loss ignores padding tokens.
    """

    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Stack input features (already fixed-size from feature extractor)
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad labels to max length in batch, using -100 for padding
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace pad token IDs with -100 for loss masking
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if the model prepends it automatically
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_compute_metrics(processor: WhisperProcessor):
    """Build a compute_metrics function for WER evaluation."""
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token for decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Filter out empty references (would cause WER computation to fail)
        pairs = [
            (p.strip(), l.strip())
            for p, l in zip(pred_str, label_str)
            if l.strip()
        ]
        if not pairs:
            return {"wer": 1.0}

        preds, labels = zip(*pairs)
        wer = wer_metric.compute(predictions=list(preds), references=list(labels))

        return {"wer": wer}

    return compute_metrics


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def setup_model_and_processor():
    """Load base model, apply LoRA, configure generation settings."""
    logger.info("Loading base model: %s", BASE_MODEL)

    # Load processor (feature extractor + tokenizer)
    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL,
        language="ar",
        task="transcribe",
    )

    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if FP16 else torch.float32,
    )

    # Anti-hallucination: configure generation
    model.generation_config = GenerationConfig(
        language="ar",
        task="transcribe",
        no_repeat_ngram_size=3,
        condition_on_previous_text=False,
        forced_decoder_ids=processor.get_decoder_prompt_ids(
            language="ar", task="transcribe"
        ),
        begin_suppress_tokens=[],
        max_length=448,
    )

    # Enable gradient checkpointing to save memory
    model.config.use_cache = False  # Required for gradient checkpointing
    model.gradient_checkpointing_enable()

    # Freeze encoder (it generalizes well) and only fine-tune decoder via LoRA
    # This is the recommended approach for Whisper fine-tuning since the
    # encoder's acoustic features transfer well across domains
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.SEQ_2_SEQ_LM,
        bias="none",
        modules_to_save=["proj_out"],  # Keep the output projection trainable
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("Whisper Fine-Tuning for Quranic Arabic")
    logger.info("=" * 60)
    logger.info("Base model:     %s", BASE_MODEL)
    logger.info("LoRA rank:      %d (alpha=%d)", LORA_R, LORA_ALPHA)
    logger.info("Epochs:         %d", NUM_EPOCHS)
    logger.info("Learning rate:  %s", LEARNING_RATE)
    logger.info("Batch size:     %d x %d = %d effective",
                PER_DEVICE_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
                PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    logger.info("Output dir:     %s", OUTPUT_DIR)
    logger.info("Hub repo:       %s", HUB_REPO)
    logger.info("")

    # Setup
    model, processor = setup_model_and_processor()
    train_dataset, val_dataset = load_and_prepare_datasets(processor)
    data_collator = WhisperDataCollator(processor=processor)
    compute_metrics = build_compute_metrics(processor)

    # HuggingFace Hub token
    hf_token = os.getenv("HF_TOKEN")
    push_to_hub = hf_token is not None
    if push_to_hub:
        logger.info("HF_TOKEN found — will push to Hub: %s", HUB_REPO)
    else:
        logger.info("HF_TOKEN not set — model will only be saved locally")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        run_name="whisper-small-quran-lora",

        # Training schedule
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",

        # Batch size
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE * 2,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        # Precision and memory
        fp16=FP16,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        # Evaluation
        eval_strategy="steps",
        eval_steps=500,
        predict_with_generate=True,
        generation_max_length=448,

        # Checkpointing
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,

        # Logging
        logging_strategy="steps",
        logging_steps=50,
        report_to=["tensorboard"],

        # Hub
        push_to_hub=push_to_hub,
        hub_model_id=HUB_REPO if push_to_hub else None,
        hub_token=hf_token,
        hub_strategy="checkpoint" if push_to_hub else "every_save",
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Log results
    logger.info("Training complete!")
    logger.info("Training loss: %.4f", train_result.training_loss)

    # Evaluate
    if val_dataset is not None:
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        logger.info("Validation WER: %.4f", eval_results.get("eval_wer", -1))

    # Save the LoRA adapter
    logger.info("Saving LoRA adapter to %s", OUTPUT_DIR / "lora-adapter")
    model.save_pretrained(str(OUTPUT_DIR / "lora-adapter"))

    # Merge LoRA weights into the base model and save the full model
    logger.info("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(OUTPUT_DIR / "merged"))
    processor.save_pretrained(str(OUTPUT_DIR / "merged"))

    # Configure generation_config.json for the merged model
    # This is critical for transformers.js — without forced_decoder_ids and
    # lang_to_id, the browser inference will crash
    gen_config = merged_model.generation_config
    gen_config.language = "ar"
    gen_config.task = "transcribe"
    gen_config.no_repeat_ngram_size = 3
    gen_config.condition_on_previous_text = False
    gen_config.save_pretrained(str(OUTPUT_DIR / "merged"))

    logger.info("Merged model saved to %s", OUTPUT_DIR / "merged")

    # Push to Hub
    if push_to_hub:
        logger.info("Pushing merged model to Hub: %s", HUB_REPO)
        merged_model.push_to_hub(HUB_REPO, token=hf_token)
        processor.push_to_hub(HUB_REPO, token=hf_token)
        logger.info("Model pushed to https://huggingface.co/%s", HUB_REPO)

    # Save training metrics
    metrics = {
        "base_model": BASE_MODEL,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "effective_batch_size": PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "training_loss": train_result.training_loss,
    }
    if val_dataset is not None:
        metrics["eval_wer"] = eval_results.get("eval_wer", -1)

    metrics_path = OUTPUT_DIR / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("  1. Export to ONNX:  python scripts/export_whisper_onnx.py")
    logger.info("  2. Copy to web app: cp -r output/whisper-small-quran/onnx-export/* web/frontend/public/models/whisper-quran/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
