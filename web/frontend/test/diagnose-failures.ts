#!/usr/bin/env npx tsx
/** Diagnose the 4 fixable streaming failures */
import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";
import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { CTCDecoder } from "../src/worker/ctc-decode.ts";
import { QuranDB, normalizeArabic } from "../src/lib/quran-db.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");

async function main() {
  const session = await ort.InferenceSession.create(
    resolve(ROOT, "public/fastconformer_ar_ctc_q8.onnx"), { executionProviders: ["cpu"] });
  const decoder = new CTCDecoder(JSON.parse(readFileSync(resolve(ROOT, "public/vocab.json"), "utf-8")));
  const db = new QuranDB(JSON.parse(readFileSync(resolve(ROOT, "public/quran.json"), "utf-8")));
  db.loadDisambiguationMap(JSON.parse(readFileSync(resolve(ROOT, "public/ambiguity-compact.json"), "utf-8")));

  const targets = ["tarteel_020", "tarteel_028", "tarteel_031", "tarteel_035"];
  const manifest = JSON.parse(readFileSync(resolve(ROOT, "../../benchmark/test_corpus_expanded/manifest.json"), "utf-8"));

  async function transcribe(audio: Float32Array) {
    const { features, timeFrames } = await computeMelSpectrogram(audio);
    const input = new ort.Tensor("float32", features, [1, 80, timeFrames]);
    const length = new ort.Tensor("int64", BigInt64Array.from([BigInt(timeFrames)]), [1]);
    const results = await session.run({ [session.inputNames[0]]: input, [session.inputNames[1]]: length });
    const out = results[session.outputNames[0]];
    const [, ts, vs] = out.dims as number[];
    return { text: decoder.decode(out.data as Float32Array, ts, vs).text, ts, vs };
  }

  for (const id of targets) {
    const s = manifest.samples.find((x: any) => x.id === id);
    if (!s) continue;
    const fpath = resolve(ROOT, "../../benchmark/test_corpus_expanded", s.file);
    const buf = execSync(`ffmpeg -hide_banner -loglevel error -i "${fpath}" -f f32le -ar 16000 -ac 1 pipe:1`, { maxBuffer: 50 * 1024 * 1024 });
    const audio = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    const durS = (audio.length / 16000).toFixed(1);

    // Full audio
    const full = await transcribe(audio);
    const fullNorm = normalizeArabic(full.text);

    // 4s window
    const audio4s = audio.slice(0, 16000 * 4);
    const t4s = await transcribe(audio4s);
    const norm4s = normalizeArabic(t4s.text);

    // 6s window
    const audio6s = audio.slice(0, Math.min(16000 * 6, audio.length));
    const t6s = await transcribe(audio6s);
    const norm6s = normalizeArabic(t6s.text);

    const matchFull = db.matchVerse(fullNorm, 0.25, 4, null, 10, null);
    const match4s = db.matchVerse(norm4s, 0.25, 4, null, 10, null);
    const match6s = db.matchVerse(norm6s, 0.25, 4, null, 10, null);

    const exp = s.expected_verses[0];
    const correctVerse = db.getVerse(exp.surah, exp.ayah);

    console.log(`\n=== ${id} (expected ${exp.surah}:${exp.ayah}, duration ${durS}s) ===`);
    console.log(`Expected: ${correctVerse?.text_clean?.slice(0, 80)}`);
    console.log(`Full transcript (${durS}s): ${fullNorm.slice(0, 80)}`);
    console.log(`4s transcript:             ${norm4s.slice(0, 80)}`);
    console.log(`6s transcript:             ${norm6s.slice(0, 80)}`);
    console.log(`Full match: ${matchFull ? `${matchFull.surah}:${matchFull.ayah} score=${matchFull.score.toFixed(3)}` : "null"}`);
    console.log(`4s match:   ${match4s ? `${match4s.surah}:${match4s.ayah} score=${match4s.score.toFixed(3)}` : "null"}`);
    console.log(`6s match:   ${match6s ? `${match6s.surah}:${match6s.ayah} score=${match6s.score.toFixed(3)}` : "null"}`);

    // Runners-up for 4s
    if (match4s?.runners_up) {
      console.log("4s runners-up:");
      for (const ru of match4s.runners_up.slice(0, 5)) {
        const isCorrect = ru.surah === exp.surah && ru.ayah === exp.ayah ? " ← CORRECT" : "";
        console.log(`  ${ru.surah}:${ru.ayah} score=${ru.score.toFixed(3)}${isCorrect}`);
      }
    }
  }
}

main().catch(e => { console.error(e); process.exit(1); });
