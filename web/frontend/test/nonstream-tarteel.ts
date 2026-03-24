#!/usr/bin/env npx tsx
/** Quick non-streaming benchmark on Tarteel expanded corpus */
import { execSync } from "node:child_process";
import { readFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";
import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { CTCDecoder } from "../src/worker/ctc-decode.ts";
import { QuranDB, normalizeArabic } from "../src/lib/quran-db.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const LIMIT = parseInt(process.argv.find(a => a.startsWith("--sample="))?.split("=")[1] ?? "60");

async function main() {
  const session = await ort.InferenceSession.create(
    resolve(ROOT, "public/fastconformer_ar_ctc_q8.onnx"),
    { executionProviders: ["cpu"] },
  );
  const decoder = new CTCDecoder(JSON.parse(readFileSync(resolve(ROOT, "public/vocab.json"), "utf-8")));
  const db = new QuranDB(JSON.parse(readFileSync(resolve(ROOT, "public/quran.json"), "utf-8")));
  db.loadDisambiguationMap(JSON.parse(readFileSync(resolve(ROOT, "public/ambiguity-compact.json"), "utf-8")));

  const manifest = JSON.parse(readFileSync(resolve(ROOT, "../../benchmark/test_corpus_expanded/manifest.json"), "utf-8"));
  let correct = 0, total = 0, noMatch = 0;
  const failures: string[] = [];

  for (const s of manifest.samples.slice(0, LIMIT)) {
    const fpath = resolve(ROOT, "../../benchmark/test_corpus_expanded", s.file);
    if (!existsSync(fpath)) continue;

    const buf = execSync(
      `ffmpeg -hide_banner -loglevel error -i "${fpath}" -f f32le -ar 16000 -ac 1 pipe:1`,
      { maxBuffer: 50 * 1024 * 1024 },
    );
    const audio = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    const { features, timeFrames } = await computeMelSpectrogram(audio);
    const input = new ort.Tensor("float32", features, [1, 80, timeFrames]);
    const length = new ort.Tensor("int64", BigInt64Array.from([BigInt(timeFrames)]), [1]);
    const results = await session.run({ [session.inputNames[0]]: input, [session.inputNames[1]]: length });
    const out = results[session.outputNames[0]];
    const [, ts, vs] = out.dims as number[];
    const { text } = decoder.decode(out.data as Float32Array, ts, vs);
    const norm = normalizeArabic(text);
    const match = db.matchVerse(norm, 0.25, 4, null, 10, null);

    const exp = s.expected_verses[0];
    const ok = match && match.surah === exp.surah && match.ayah === exp.ayah;
    if (ok) correct++;
    else {
      if (!match) noMatch++;
      failures.push(`  ${s.id}: ${exp.surah}:${exp.ayah} → ${match ? `${match.surah}:${match.ayah} (${match.score.toFixed(3)})` : "?"}`);
    }
    total++;
    process.stdout.write(ok ? "." : "✗");
    if (total % 20 === 0) process.stdout.write(` ${correct}/${total}\n`);
  }

  console.log(`\n\nNon-streaming Tarteel: ${correct}/${total} (${(correct / total * 100).toFixed(1)}%)`);
  console.log(`No match: ${noMatch}`);
  if (failures.length) {
    console.log(`\nFailures (${failures.length}):`);
    for (const f of failures) console.log(f);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
