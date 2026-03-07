#!/usr/bin/env python3
"""
Lightweight export of NVIDIA NeMo FastConformer to ONNX (CTC-only).

Bypasses the full NeMo toolkit dependency chain by directly extracting
the PyTorch checkpoint from the .nemo archive and manually constructing
the export.

Requirements:
  pip install torch onnx onnxruntime sentencepiece huggingface-hub pyyaml numpy

Usage:
  python export_nvidia_model.py [--output-dir ./public]
"""

import argparse
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import yaml
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def parse_args():
    parser = argparse.ArgumentParser(description="Export NVIDIA NeMo FastConformer to ONNX (lightweight)")
    parser.add_argument("--output-dir", type=str, default="./public")
    parser.add_argument("--model-name", type=str, default="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0")
    parser.add_argument("--skip-quantization", action="store_true")
    parser.add_argument("--nemo-file", type=str, default=None, help="Path to existing .nemo file (skip download)")
    return parser.parse_args()


def download_model(model_id: str, cache_dir: Path) -> Path:
    """Download .nemo file from HuggingFace."""
    from huggingface_hub import hf_hub_download, list_repo_files

    print(f"[1/6] Downloading {model_id}...")

    # List files to find .nemo file
    files = list_repo_files(model_id)
    nemo_files = [f for f in files if f.endswith(".nemo")]
    print(f"  Found files: {nemo_files}")

    if not nemo_files:
        raise ValueError(f"No .nemo file found in {model_id}. Files: {files[:20]}")

    nemo_path = hf_hub_download(
        repo_id=model_id,
        filename=nemo_files[0],
        cache_dir=str(cache_dir),
        resume_download=True,
    )
    print(f"  Downloaded: {nemo_path}")
    return Path(nemo_path)


def extract_nemo_archive(nemo_path: Path, extract_dir: Path) -> dict:
    """Extract .nemo archive and return config + paths."""
    print("[2/6] Extracting .nemo archive...")

    with tarfile.open(nemo_path, "r:*") as tar:
        tar.extractall(extract_dir, filter="data")

    # Find config and checkpoint
    config_path = None
    ckpt_path = None
    sp_model_path = None

    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            full = os.path.join(root, f)
            if f == "model_config.yaml":
                config_path = full
            elif f.endswith(".ckpt") or f == "model_weights.ckpt":
                ckpt_path = full
            elif f.endswith(".model") and "tokenizer" in full.lower():
                sp_model_path = full

    if not config_path:
        raise FileNotFoundError("model_config.yaml not found in .nemo archive")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"  Config: {config_path}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Tokenizer: {sp_model_path}")

    return {
        "config": config,
        "ckpt_path": ckpt_path,
        "sp_model_path": sp_model_path,
        "extract_dir": extract_dir,
    }


def validate_mel_config(config: dict):
    """Validate and print mel config comparison."""
    print("[3/6] Validating mel spectrogram config...")

    preprocessor = config.get("preprocessor", {})

    expected = {
        "sample_rate": 16000,
        "n_fft": 512,
        "hop_length": 160,
        "win_length": 400,
        "features (n_mels)": 80,
    }

    # NeMo config uses different key names
    actual = {
        "sample_rate": preprocessor.get("sample_rate", "?"),
        "n_fft": preprocessor.get("n_fft", "?"),
        "hop_length": preprocessor.get("hop_length", "?"),
        "win_length": preprocessor.get("win_length", "?"),
        "features (n_mels)": preprocessor.get("features", "?"),
    }

    print(f"  {'Parameter':<25} {'Expected':<12} {'Actual':<12} {'Match'}")
    print("  " + "-" * 60)
    mismatches = []
    for k in expected:
        match = expected[k] == actual[k]
        mark = "Y" if match else "N"
        print(f"  {k:<25} {str(expected[k]):<12} {str(actual[k]):<12} {mark}")
        if not match and actual[k] != "?":
            mismatches.append((k, expected[k], actual[k]))

    # Also print extra NeMo-specific config
    extra_keys = ["preemph", "dither", "normalize", "window", "log_zero_guard_type", "log_zero_guard_value"]
    for ek in extra_keys:
        val = preprocessor.get(ek, "?")
        print(f"  {ek:<25} {'—':<12} {str(val):<12}")

    if mismatches:
        print(f"\n  WARNING: {len(mismatches)} mismatch(es) — you MUST update mel.ts!")
        for k, exp, act in mismatches:
            print(f"    {k}: {exp} -> {act}")
    else:
        print("\n  All critical mel params match.")

    return preprocessor


def export_onnx_nemo_native(nemo_path: Path, output_dir: Path) -> Path:
    """
    Try using NeMo's built-in export if available.
    Falls back to manual PyTorch export.
    """
    print("[4/6] Exporting to ONNX...")

    onnx_fp32 = output_dir / "fastconformer_nvidia_ctc_fp32.onnx"

    # Try NeMo native export first
    try:
        # NeMo 2.x: try minimal import
        os.environ.setdefault("NEMO_TESTING", "1")  # suppress some imports
        import nemo.collections.asr as nemo_asr

        print("  Using NeMo native export...")
        model = nemo_asr.models.ASRModel.restore_from(str(nemo_path), map_location="cpu")

        # For hybrid models, configure CTC-only export
        if hasattr(model, "set_export_config"):
            model.set_export_config({"decoder_type": "ctc"})
            print("  Set decoder_type=ctc")

        model.export(str(onnx_fp32))
        print(f"  Exported: {onnx_fp32}")
        return onnx_fp32

    except Exception as e:
        print(f"  NeMo native export failed: {e}")
        print("  Falling back to manual PyTorch export...")

    # Manual export: load checkpoint, build model, torch.onnx.export
    return _manual_export(nemo_path, onnx_fp32)


def _manual_export(nemo_path: Path, onnx_path: Path) -> Path:
    """Manual ONNX export by extracting PyTorch weights."""

    with tempfile.TemporaryDirectory() as tmpdir:
        info = extract_nemo_archive(nemo_path, Path(tmpdir))
        config = info["config"]
        ckpt_path = info["ckpt_path"]

        if not ckpt_path:
            raise FileNotFoundError("No .ckpt found in .nemo archive")

        # Load checkpoint
        print("  Loading checkpoint...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # The checkpoint contains state_dict for the full hybrid model.
        # We need: encoder + CTC decoder (not RNNT decoder).
        # NeMo typically stores as 'state_dict' or directly as the dict.
        state_dict = ckpt.get("state_dict", ckpt)

        # Print model structure summary
        prefixes = set()
        for k in state_dict.keys():
            parts = k.split(".")
            if len(parts) >= 2:
                prefixes.add(f"{parts[0]}.{parts[1]}")
        print(f"  State dict prefixes: {sorted(prefixes)[:20]}")
        print(f"  Total parameters: {len(state_dict)}")

        # We need to reconstruct the model architecture from config.
        # This is complex for FastConformer. Instead, let's use a simpler approach:
        # Use torch.jit.trace with the NeMo model directly.

        # Actually, the most reliable approach for hybrid models is to use
        # the sherpa-onnx export tools which handle this case well.
        print("\n  The manual export path requires reconstructing the FastConformer")
        print("  architecture from scratch, which is complex.")
        print("\n  RECOMMENDED: Use sherpa-onnx export tools instead:")
        print("    pip install sherpa-onnx")
        print("    python -m sherpa_onnx.export --framework nemo \\")
        print(f"        --type ctc --nemo {nemo_path} --output {onnx_path}")
        print("\n  OR use the NeMo Docker container:")
        print("    docker run --rm -v $(pwd):/workspace nvcr.io/nvidia/nemo:25.01 \\")
        print("      python -c \"")
        print("        import nemo.collections.asr as nemo_asr")
        print(f"        m = nemo_asr.models.ASRModel.restore_from('/workspace/{nemo_path.name}')")
        print("        m.set_export_config({\\\"decoder_type\\\": \\\"ctc\\\"})")
        print(f"        m.export('/workspace/{onnx_path.name}')\"")

        raise RuntimeError("Manual export not implemented — use sherpa-onnx or NeMo Docker")


def extract_vocabulary(nemo_path: Path, output_dir: Path) -> Path:
    """Extract vocabulary from .nemo archive."""
    print("[5/6] Extracting vocabulary...")

    import sentencepiece as spm

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract .nemo
        with tarfile.open(nemo_path, "r:*") as tar:
            tar.extractall(tmpdir, filter="data")

        # Find SentencePiece model
        sp_path = None
        for root, dirs, files in os.walk(tmpdir):
            for f in files:
                if f.endswith(".model") and "tokenizer" in os.path.join(root, f).lower():
                    sp_path = os.path.join(root, f)
                    break

        if not sp_path:
            # Try any .model file
            for root, dirs, files in os.walk(tmpdir):
                for f in files:
                    if f.endswith(".model"):
                        sp_path = os.path.join(root, f)
                        break

        if not sp_path:
            print("  No SentencePiece model found!")
            return None

        print(f"  Found tokenizer: {sp_path}")

        # Load SentencePiece
        sp = spm.SentencePieceProcessor()
        sp.load(sp_path)

        vocab_size = sp.get_piece_size()
        print(f"  SentencePiece vocab size: {vocab_size}")

        # Build vocab JSON in our format: {"0": "token", "1": "token", ...}
        vocab = {}
        for i in range(vocab_size):
            piece = sp.id_to_piece(i)
            vocab[str(i)] = piece

        # Check for blank token
        has_blank = any(v in ("<blank>", "<blk>") for v in vocab.values())
        if not has_blank:
            # Add blank as last token (standard CTC convention)
            blank_id = vocab_size
            vocab[str(blank_id)] = "<blank>"
            print(f"  Added <blank> token at index {blank_id}")

        # Print sample tokens
        print(f"  Sample tokens: {dict(list(vocab.items())[:10])}")
        print(f"  Last tokens: {dict(list(vocab.items())[-5:])}")

        # Find blank ID
        blank_id = None
        for k, v in vocab.items():
            if v in ("<blank>", "<blk>"):
                blank_id = int(k)
                break
        print(f"  Blank token ID: {blank_id}")

        # Save
        vocab_path = output_dir / "vocab_nvidia.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        print(f"  Saved: {vocab_path} ({len(vocab)} tokens)")
        return vocab_path


def quantize_model(onnx_path: Path, output_dir: Path) -> Path:
    """Quantize to int8."""
    print("[6/6] Quantizing to int8...")

    q8_path = output_dir / "fastconformer_nvidia_ctc_q8.onnx"

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(q8_path),
        weight_type=QuantType.QInt8,
        optimize_model=True,
    )

    fp32_mb = onnx_path.stat().st_size / (1024 * 1024)
    q8_mb = q8_path.stat().st_size / (1024 * 1024)
    print(f"  FP32: {fp32_mb:.1f} MB -> INT8: {q8_mb:.1f} MB ({(1-q8_mb/fp32_mb)*100:.0f}% smaller)")
    return q8_path


def test_inference(model_path: Path, vocab_path: Path):
    """Quick inference test."""
    print("[Test] Running inference test...")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path))
    inputs = {i.name: i for i in sess.get_inputs()}
    outputs = {o.name: o for o in sess.get_outputs()}

    print(f"  Inputs: {[(n, i.shape) for n, i in inputs.items()]}")
    print(f"  Outputs: {[(n, o.shape) for n, o in outputs.items()]}")

    # Dummy inference
    inp_names = list(inputs.keys())
    dummy = np.random.randn(1, 80, 100).astype(np.float32)
    feeds = {inp_names[0]: dummy}
    if len(inp_names) > 1:
        feeds[inp_names[1]] = np.array([100], dtype=np.int64)

    result = sess.run(None, feeds)
    shape = result[0].shape
    print(f"  Output shape: {shape}")

    if len(shape) == 3:
        print(f"  [batch={shape[0]}, time={shape[1]}, vocab={shape[2]}]")
    print("  Inference test passed!")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("NVIDIA FastConformer -> ONNX Export")
    print("=" * 70)

    # Step 1: Get .nemo file
    if args.nemo_file:
        nemo_path = Path(args.nemo_file)
        print(f"[1/6] Using existing .nemo: {nemo_path}")
    else:
        nemo_path = download_model(args.model_name, cache_dir)

    # Step 2-3: Extract config and validate mel
    with tempfile.TemporaryDirectory() as tmpdir:
        info = extract_nemo_archive(nemo_path, Path(tmpdir))
        mel_config = validate_mel_config(info["config"])

    # Step 4: Export ONNX
    try:
        onnx_path = export_onnx_nemo_native(nemo_path, output_dir)
    except Exception as e:
        print(f"\n  Export failed: {e}")
        print("\n  Trying sherpa-onnx approach...")

        # Try sherpa-onnx
        try:
            import subprocess
            pip_result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "sherpa-onnx"],
                capture_output=True, text=True
            )
            if pip_result.returncode == 0:
                onnx_path = output_dir / "fastconformer_nvidia_ctc_fp32.onnx"
                result = subprocess.run(
                    [sys.executable, "-m", "sherpa_onnx.export",
                     "--framework", "nemo", "--type", "ctc",
                     "--nemo", str(nemo_path), "--output", str(onnx_path)],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    raise RuntimeError(f"sherpa-onnx export failed: {result.stderr}")
                print(f"  Exported via sherpa-onnx: {onnx_path}")
            else:
                raise RuntimeError("Could not install sherpa-onnx")
        except Exception as e2:
            print(f"\n  All export methods failed.")
            print(f"  Error: {e2}")
            print(f"\n  Please export manually using NeMo Docker:")
            print(f"    See MODELS_RESEARCH.md for instructions")

            # Still extract vocabulary
            vocab_path = extract_vocabulary(nemo_path, output_dir)
            if vocab_path:
                print(f"\n  Vocabulary extracted to: {vocab_path}")
                print("  You can still integrate the model once you export the ONNX file.")
            return

    # Step 5: Extract vocabulary
    vocab_path = extract_vocabulary(nemo_path, output_dir)

    # Step 6: Quantize
    if args.skip_quantization:
        final_path = onnx_path
    else:
        final_path = quantize_model(onnx_path, output_dir)

    # Test
    if vocab_path:
        test_inference(final_path, vocab_path)

    print("\n" + "=" * 70)
    print("DONE! Next steps:")
    print(f"  1. Model: {final_path}")
    print(f"  2. Vocab: {vocab_path}")
    print(f"  3. Update inference.ts MODEL_URL")
    print(f"  4. Replace or rename vocab.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
