import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { QuranDB } from "../../src/lib/quran-db.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));

let _fixtureDb: QuranDB | null = null;
let _fullDb: QuranDB | null = null;

export function getFixtureQuranDB(): QuranDB {
  if (!_fixtureDb) {
    const data = JSON.parse(
      readFileSync(resolve(__dirname, "../fixtures/quran-sample.json"), "utf-8")
    );
    _fixtureDb = new QuranDB(data);
  }
  return _fixtureDb;
}

export function getFullQuranDB(): QuranDB {
  if (!_fullDb) {
    const data = JSON.parse(
      readFileSync(resolve(__dirname, "../../public/quran.json"), "utf-8")
    );
    _fullDb = new QuranDB(data);
  }
  return _fullDb;
}
