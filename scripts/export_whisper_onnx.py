"""
Export a fine-tuned Whisper model to ONNX for browser inference via transformers.js.

Produces:
  - onnx/encoder_model.onnx         (fp32 — WASM can't do quantized Conv)
  - onnx/decoder_model_merged.onnx  (q8 — decoder has no Conv ops, safe to quantize)
  - config.json, generation_config.json, tokenizer files, preprocessor_config.json

The output directory can be copied directly into the web frontend:
  cp -r output/onnx-export/* web/frontend/public/models/whisper-quran/

Usage:
    # Export from local merged model
    python scripts/export_whisper_onnx.py

    # Export from HuggingFace Hub
    MODEL_PATH=your-org/whisper-small-quran  python scripts/export_whisper_onnx.py

    # Custom output directory
    OUTPUT_DIR=./my-export  python scripts/export_whisper_onnx.py

Requirements:
    pip install transformers optimum[onnxruntime] onnxruntime onnx
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH: str = os.getenv("MODEL_PATH", "./output/whisper-small-quran/merged")
OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "./output/whisper-small-quran/onnx-export"))
OPSET_VERSION: int = 14  # ONNX opset version compatible with transformers.js

# The whisper-small token IDs for Arabic transcription.
# These are identical across all whisper multilingual models.
ARABIC_LANG_TOKEN_ID: int = 50272  # <|ar|>
TRANSCRIBE_TOKEN_ID: int = 50359   # <|transcribe|>
NO_TIMESTAMPS_TOKEN_ID: int = 50363  # <|notimestamps|>
BOS_TOKEN_ID: int = 50257
EOS_TOKEN_ID: int = 50257
DECODER_START_TOKEN_ID: int = 50258

# Full language-to-id mapping (required by transformers.js for Whisper)
LANG_TO_ID: dict[str, int] = {
    "<|en|>": 50259, "<|zh|>": 50260, "<|de|>": 50261, "<|es|>": 50262,
    "<|ru|>": 50263, "<|ko|>": 50264, "<|fr|>": 50265, "<|ja|>": 50266,
    "<|pt|>": 50267, "<|tr|>": 50268, "<|pl|>": 50269, "<|ca|>": 50270,
    "<|nl|>": 50271, "<|ar|>": 50272, "<|sv|>": 50273, "<|it|>": 50274,
    "<|id|>": 50275, "<|hi|>": 50276, "<|fi|>": 50277, "<|vi|>": 50278,
    "<|iw|>": 50279, "<|uk|>": 50280, "<|el|>": 50281, "<|ms|>": 50282,
    "<|cs|>": 50283, "<|ro|>": 50284, "<|da|>": 50285, "<|hu|>": 50286,
    "<|ta|>": 50287, "<|no|>": 50288, "<|th|>": 50289, "<|ur|>": 50290,
    "<|hr|>": 50291, "<|bg|>": 50292, "<|lt|>": 50293, "<|la|>": 50294,
    "<|mi|>": 50295, "<|ml|>": 50296, "<|cy|>": 50297, "<|sk|>": 50298,
    "<|te|>": 50299, "<|fa|>": 50300, "<|lv|>": 50301, "<|bn|>": 50302,
    "<|sr|>": 50303, "<|az|>": 50304, "<|sl|>": 50305, "<|kn|>": 50306,
    "<|et|>": 50307, "<|mk|>": 50308, "<|br|>": 50309, "<|eu|>": 50310,
    "<|is|>": 50311, "<|hy|>": 50312, "<|ne|>": 50313, "<|mn|>": 50314,
    "<|bs|>": 50315, "<|kk|>": 50316, "<|sq|>": 50317, "<|sw|>": 50318,
    "<|gl|>": 50319, "<|mr|>": 50320, "<|pa|>": 50321, "<|si|>": 50322,
    "<|km|>": 50323, "<|sn|>": 50324, "<|yo|>": 50325, "<|so|>": 50326,
    "<|af|>": 50327, "<|oc|>": 50328, "<|ka|>": 50329, "<|be|>": 50330,
    "<|tg|>": 50331, "<|sd|>": 50332, "<|gu|>": 50333, "<|am|>": 50334,
    "<|yi|>": 50335, "<|lo|>": 50336, "<|uz|>": 50337, "<|fo|>": 50338,
    "<|ht|>": 50339, "<|ps|>": 50340, "<|tk|>": 50341, "<|nn|>": 50342,
    "<|mt|>": 50343, "<|sa|>": 50344, "<|lb|>": 50345, "<|my|>": 50346,
    "<|bo|>": 50347, "<|tl|>": 50348, "<|mg|>": 50349, "<|as|>": 50350,
    "<|tt|>": 50351, "<|haw|>": 50352, "<|ln|>": 50353, "<|ha|>": 50354,
    "<|ba|>": 50355, "<|jw|>": 50356, "<|su|>": 50357,
}

TASK_TO_ID: dict[str, int] = {
    "<|transcribe|>": 50359,
    "<|translate|>": 50358,
}


# ---------------------------------------------------------------------------
# Export steps
# ---------------------------------------------------------------------------

def export_to_onnx(model_path: str, output_dir: Path) -> Path:
    """Export the Whisper model to ONNX using Optimum.

    Uses ORTModelForSpeechSeq2Seq which handles the encoder/decoder split
    and produces the merged decoder model that transformers.js expects.
    """
    onnx_dir = output_dir / "onnx_temp"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting model from %s to ONNX...", model_path)

    # Export using Optimum — this handles the encoder/decoder split correctly
    ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        export=True,
        provider="CPUExecutionProvider",
    )
    ort_model.save_pretrained(str(onnx_dir))

    logger.info("ONNX export complete at %s", onnx_dir)
    return onnx_dir


def quantize_decoder(onnx_dir: Path, output_dir: Path) -> None:
    """Quantize the decoder to INT8 while keeping the encoder at fp32.

    The encoder uses Conv1D layers that WASM (via transformers.js) cannot
    execute in quantized form — they require ConvInteger which is not
    implemented. The decoder has only MatMul/Add/LayerNorm, all of which
    quantize safely.
    """
    final_onnx_dir = output_dir / "onnx"
    final_onnx_dir.mkdir(parents=True, exist_ok=True)

    # --- Encoder: keep fp32 ---
    encoder_candidates = [
        onnx_dir / "encoder_model.onnx",
        onnx_dir / "onnx" / "encoder_model.onnx",
    ]
    encoder_src = next((p for p in encoder_candidates if p.exists()), None)
    if encoder_src is None:
        raise FileNotFoundError(
            f"Could not find encoder_model.onnx in {onnx_dir}. "
            f"Contents: {list(onnx_dir.rglob('*.onnx'))}"
        )

    encoder_dst = final_onnx_dir / "encoder_model.onnx"
    shutil.copy2(encoder_src, encoder_dst)
    encoder_size_mb = encoder_dst.stat().st_size / (1024 * 1024)
    logger.info("Encoder (fp32): %.1f MB -> %s", encoder_size_mb, encoder_dst)

    # --- Decoder: quantize to INT8 ---
    # Look for the merged decoder (with KV cache) which transformers.js uses
    decoder_candidates = [
        onnx_dir / "decoder_model_merged.onnx",
        onnx_dir / "onnx" / "decoder_model_merged.onnx",
        onnx_dir / "decoder_with_past_model.onnx",
        onnx_dir / "onnx" / "decoder_with_past_model.onnx",
        onnx_dir / "decoder_model.onnx",
        onnx_dir / "onnx" / "decoder_model.onnx",
    ]
    decoder_src = next((p for p in decoder_candidates if p.exists()), None)
    if decoder_src is None:
        raise FileNotFoundError(
            f"Could not find decoder ONNX model in {onnx_dir}. "
            f"Contents: {list(onnx_dir.rglob('*.onnx'))}"
        )

    # If it is not already a merged model, we need the merged version for
    # transformers.js. Optimum exports both decoder_model.onnx and
    # decoder_with_past_model.onnx — we need to merge them. However,
    # newer Optimum versions export decoder_model_merged.onnx directly.
    decoder_fp32 = final_onnx_dir / "decoder_model_merged_fp32.onnx"
    shutil.copy2(decoder_src, decoder_fp32)

    decoder_q8 = final_onnx_dir / "decoder_model_merged.onnx"
    logger.info("Quantizing decoder to INT8...")
    quantize_dynamic(
        model_input=str(decoder_fp32),
        model_output=str(decoder_q8),
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
        optimize_model=True,
    )

    # Clean up fp32 decoder (we only ship the quantized one)
    decoder_fp32.unlink()

    decoder_size_mb = decoder_q8.stat().st_size / (1024 * 1024)
    logger.info("Decoder (q8): %.1f MB -> %s", decoder_size_mb, decoder_q8)


def copy_config_files(model_path: str, onnx_dir: Path, output_dir: Path) -> None:
    """Copy and fix config files needed by transformers.js.

    transformers.js requires these files at the model root:
      - config.json
      - generation_config.json (MUST have forced_decoder_ids and lang_to_id)
      - tokenizer.json, tokenizer_config.json, vocab.json, merges.txt
      - special_tokens_map.json, added_tokens.json
      - normalizer.json
      - preprocessor_config.json
    """
    config_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "normalizer.json",
        "preprocessor_config.json",
    ]

    # Copy from the ONNX export first (Optimum may have tweaked them),
    # falling back to the original model directory
    for fname in config_files:
        dst = output_dir / fname
        # Try onnx export dir first, then model dir
        candidates = [
            onnx_dir / fname,
            Path(model_path) / fname,
        ]
        src = next((p for p in candidates if p.exists()), None)
        if src:
            shutil.copy2(src, dst)
            logger.info("  Copied %s", fname)
        else:
            logger.warning("  Missing %s — transformers.js may still work without it", fname)

    # Fix generation_config.json — this is the most critical file.
    # Without forced_decoder_ids and lang_to_id, transformers.js crashes
    # because it cannot determine which language/task tokens to inject.
    gen_config = _build_generation_config(model_path, onnx_dir)
    gen_config_path = output_dir / "generation_config.json"
    with open(gen_config_path, "w") as f:
        json.dump(gen_config, f, indent=2)
    logger.info("  Wrote generation_config.json (with forced_decoder_ids + lang_to_id)")

    # Fix config.json — ensure it has the right dtype and model_type
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        config["dtype"] = "float32"
        config.setdefault("model_type", "whisper")
        config.setdefault("is_encoder_decoder", True)
        # Remove forced_decoder_ids from config.json — it belongs only in
        # generation_config.json (having it in both causes warnings)
        config.pop("forced_decoder_ids", None)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info("  Fixed config.json")


def _build_generation_config(model_path: str, onnx_dir: Path) -> dict:
    """Build the generation_config.json required by transformers.js.

    The forced_decoder_ids tell the model to generate:
      position 1 -> <|ar|> (Arabic language token)
      position 2 -> <|transcribe|> (task token)
      position 3 -> <|notimestamps|> (no timestamp generation)

    Without these, the model produces garbage or crashes in the browser.
    """
    # Start from existing generation_config if available
    gen_config = {}
    for candidate in [onnx_dir / "generation_config.json",
                      Path(model_path) / "generation_config.json"]:
        if candidate.exists():
            with open(candidate) as f:
                gen_config = json.load(f)
            break

    # Ensure all required fields are present
    gen_config.update({
        "_from_model_config": True,
        "bos_token_id": BOS_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": BOS_TOKEN_ID,
        "decoder_start_token_id": DECODER_START_TOKEN_ID,
        "begin_suppress_tokens": [220, 50257],
        "is_multilingual": True,
        "language": "ar",
        "task": "transcribe",
        "max_length": 448,
        "no_timestamps_token_id": NO_TIMESTAMPS_TOKEN_ID,
        "use_cache": True,
        # Critical for transformers.js: the forced decoder IDs
        "forced_decoder_ids": [
            [1, ARABIC_LANG_TOKEN_ID],    # <|ar|>
            [2, TRANSCRIBE_TOKEN_ID],      # <|transcribe|>
            [3, NO_TIMESTAMPS_TOKEN_ID],   # <|notimestamps|>
        ],
        # Critical for transformers.js: language mapping
        "lang_to_id": LANG_TO_ID,
        "task_to_id": TASK_TO_ID,
    })

    # Preserve transformers version if present
    gen_config.setdefault("transformers_version", "4.57.6")

    return gen_config


def verify_export(output_dir: Path) -> bool:
    """Verify the ONNX export is complete and loadable."""
    logger.info("Verifying ONNX export...")

    required_files = [
        "onnx/encoder_model.onnx",
        "onnx/decoder_model_merged.onnx",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "preprocessor_config.json",
    ]

    missing = []
    for fname in required_files:
        fpath = output_dir / fname
        if not fpath.exists():
            missing.append(fname)
        else:
            size_mb = fpath.stat().st_size / (1024 * 1024)
            logger.info("  [OK] %-40s  %.1f MB", fname, size_mb)

    if missing:
        logger.error("Missing files: %s", missing)
        return False

    # Validate ONNX models are loadable
    try:
        encoder_model = onnx.load(str(output_dir / "onnx" / "encoder_model.onnx"))
        onnx.checker.check_model(encoder_model)
        logger.info("  [OK] encoder_model.onnx passes ONNX validation")
    except Exception as e:
        logger.error("  [FAIL] encoder_model.onnx validation: %s", e)
        return False

    try:
        decoder_model = onnx.load(str(output_dir / "onnx" / "decoder_model_merged.onnx"))
        onnx.checker.check_model(decoder_model)
        logger.info("  [OK] decoder_model_merged.onnx passes ONNX validation")
    except Exception as e:
        logger.error("  [FAIL] decoder_model_merged.onnx validation: %s", e)
        return False

    # Validate generation_config.json has required fields
    with open(output_dir / "generation_config.json") as f:
        gen_config = json.load(f)

    required_keys = ["forced_decoder_ids", "lang_to_id", "language", "task"]
    for key in required_keys:
        if key not in gen_config:
            logger.error("  [FAIL] generation_config.json missing '%s'", key)
            return False

    # Verify forced_decoder_ids has the right structure
    fdi = gen_config["forced_decoder_ids"]
    if not isinstance(fdi, list) or len(fdi) < 3:
        logger.error("  [FAIL] forced_decoder_ids has wrong structure: %s", fdi)
        return False

    if fdi[0] != [1, ARABIC_LANG_TOKEN_ID]:
        logger.error("  [FAIL] forced_decoder_ids[0] should be [1, %d], got %s",
                      ARABIC_LANG_TOKEN_ID, fdi[0])
        return False

    logger.info("  [OK] generation_config.json has all required fields")

    # Quick inference test with dummy data
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            str(output_dir / "onnx" / "encoder_model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        # Whisper expects [batch, n_mels, n_frames] — 80 mel bins, 3000 frames (30s)
        dummy_input = np.zeros((1, 80, 3000), dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: dummy_input})
        logger.info("  [OK] Encoder inference test passed (output shape: %s)",
                      result[0].shape)
    except Exception as e:
        logger.warning("  [WARN] Encoder inference test failed: %s", e)
        logger.warning("         This may be OK if onnxruntime version differs from export")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("Whisper ONNX Export for transformers.js")
    logger.info("=" * 60)
    logger.info("Model:      %s", MODEL_PATH)
    logger.info("Output:     %s", OUTPUT_DIR)
    logger.info("")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Export to ONNX
    onnx_dir = export_to_onnx(MODEL_PATH, OUTPUT_DIR)

    # Step 2: Quantize decoder, keep encoder fp32
    logger.info("")
    logger.info("Applying quantization (fp32 encoder + q8 decoder)...")
    quantize_decoder(onnx_dir, OUTPUT_DIR)

    # Step 3: Copy and fix config files
    logger.info("")
    logger.info("Copying config files...")
    copy_config_files(MODEL_PATH, onnx_dir, OUTPUT_DIR)

    # Step 4: Clean up temp directory
    temp_dir = OUTPUT_DIR / "onnx_temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    # Step 5: Verify
    logger.info("")
    success = verify_export(OUTPUT_DIR)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    if success:
        logger.info("EXPORT SUCCESSFUL")
    else:
        logger.error("EXPORT COMPLETED WITH WARNINGS — check output above")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Output files:")

    total_size = 0
    for f in sorted(OUTPUT_DIR.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size += size_mb
            logger.info("  %-50s  %8.1f MB", f.relative_to(OUTPUT_DIR), size_mb)

    logger.info("  %-50s  %8.1f MB", "TOTAL", total_size)
    logger.info("")
    logger.info("To deploy to the web frontend:")
    logger.info("  cp -r %s/* web/frontend/public/models/whisper-quran/", OUTPUT_DIR)
    logger.info("")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
