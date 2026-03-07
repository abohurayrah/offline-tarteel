/**
 * Whisper ASR wrapper using @huggingface/transformers (transformers.js v3).
 *
 * Loads tarteel-ai/whisper-tiny-ar-quran ONNX model and provides a simple
 * `transcribe(audio: Float32Array) => Promise<string>` interface.
 *
 * The model is served locally from /models/whisper-quran/ and cached
 * automatically by transformers.js via the Cache API.
 */
import {
  pipeline,
  env,
  type AutomaticSpeechRecognitionOutput,
} from "@huggingface/transformers";

// Serve model from local /models/ path — no HuggingFace Hub needed at runtime
env.allowLocalModels = true;
env.allowRemoteModels = false;

const MODEL_ID = "/models/whisper-quran";

type ASRPipeline = Awaited<ReturnType<typeof pipeline<"automatic-speech-recognition">>>;
let asr: ASRPipeline | null = null;

export type ProgressCallback = (progress: {
  status: string;
  progress?: number;
  file?: string;
}) => void;

/**
 * Load the Whisper ONNX model. Call once during worker init.
 */
export async function loadWhisper(
  onProgress?: ProgressCallback,
): Promise<void> {
  asr = await pipeline("automatic-speech-recognition", MODEL_ID, {
    // fp32 encoder (31MB) — quantized encoder uses ConvInteger which WASM doesn't support
    // q8 decoder (48MB) — decoder has no Conv ops so quantization is safe
    dtype: {
      encoder_model: "fp32",
      decoder_model_merged: "q8",
    },
    device: "wasm",
    progress_callback: onProgress,
  });
}

/**
 * Transcribe a Float32Array of 16 kHz mono audio to Arabic text.
 */
export async function transcribe(audio: Float32Array): Promise<string> {
  if (!asr) throw new Error("Whisper model not loaded");

  // language=ar, task=transcribe, no timestamps are baked into generation_config.json
  // via forced_decoder_ids to avoid tokenizer lookup issues
  const result = (await asr(audio)) as AutomaticSpeechRecognitionOutput;

  return result.text.trim();
}

/**
 * Check if the model is loaded and ready.
 */
export function isReady(): boolean {
  return asr !== null;
}
