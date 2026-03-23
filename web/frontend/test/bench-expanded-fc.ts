import { execSync } from "node:child_process";
import { readFileSync, existsSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";
import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { CTCDecoder } from "../src/worker/ctc-decode.ts";
import { QuranDB } from "../src/lib/quran-db.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const CORPUS = resolve(ROOT, "../../benchmark/test_corpus_expanded");

function loadAudio(p: string): Float32Array {
  const buf = execSync(`ffmpeg -hide_banner -loglevel error -i "${p}" -f f32le -ar 16000 -ac 1 pipe:1`, { maxBuffer: 50*1024*1024 });
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

async function main() {
  console.log("Loading model...");
  const session = await ort.InferenceSession.create(resolve(ROOT, "public/fastconformer_ar_ctc_q8.onnx"), { executionProviders: ["cpu"] });
  const decoder = new CTCDecoder(JSON.parse(readFileSync(resolve(ROOT, "public/vocab.json"), "utf-8")));
  const db = new QuranDB(JSON.parse(readFileSync(resolve(ROOT, "public/quran.json"), "utf-8")));
  const manifest = JSON.parse(readFileSync(resolve(CORPUS, "manifest.json"), "utf-8"));
  
  console.log(`Running ${manifest.samples.length} samples...\n`);
  
  let correct = 0, wrong = 0, noMatch = 0;
  const cats: Record<string, [number,number]> = {};
  
  for (let i = 0; i < manifest.samples.length; i++) {
    const s = manifest.samples[i];
    const path = resolve(CORPUS, s.file);
    if (!existsSync(path)) continue;
    
    try {
      const audio = loadAudio(path);
      const { features, timeFrames } = await computeMelSpectrogram(audio);
      const input = new ort.Tensor("float32", features, [1, 80, timeFrames]);
      const length = new ort.Tensor("int64", BigInt64Array.from([BigInt(timeFrames)]), [1]);
      const results = await session.run({ [session.inputNames[0]]: input, [session.inputNames[1]]: length });
      const out = results[session.outputNames[0]];
      const [, ts, vs] = out.dims as number[];
      const { text } = decoder.decode(out.data as Float32Array, ts, vs);
      
      const match = db.matchVerse(text, 0.2, 6);
      const expected = s.expected_verses[0];
      const cat = s.category || "unknown";
      cats[cat] = cats[cat] || [0,0];
      cats[cat][1]++;
      
      let ok = false;
      if (match) {
        for (const ev of s.expected_verses) {
          if (match.surah === ev.surah && match.ayah === ev.ayah) { ok = true; break; }
        }
      }
      
      if (ok) { correct++; cats[cat][0]++; }
      else if (match) { wrong++; }
      else { noMatch++; }
      
      if ((i+1) % 50 === 0) {
        const total = correct + wrong + noMatch;
        console.log(`  [${i+1}/${manifest.samples.length}] ${correct}/${total} (${(correct/total*100).toFixed(1)}%)`);
      }
    } catch(e) { noMatch++; }
  }
  
  const total = correct + wrong + noMatch;
  console.log(`\n${"=".repeat(60)}`);
  console.log(`FASTCONFORMER + IMPROVED QURANDB on 256 Tarteel samples`);
  console.log(`${"=".repeat(60)}`);
  console.log(`Overall: ${correct}/${total} (${(correct/total*100).toFixed(1)}%)`);
  console.log(`Wrong: ${wrong}, No match: ${noMatch}`);
  for (const [cat, [ok, tot]] of Object.entries(cats)) {
    console.log(`  ${cat.padEnd(10)} ${ok}/${tot} (${(ok/tot*100).toFixed(1)}%)`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
