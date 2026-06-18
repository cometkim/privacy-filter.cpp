/// <reference lib="webworker" />
// @ts-ignore — Emscripten-generated JS module, no types
import PfModuleFactory from "../wasm/pf.js";
// @ts-ignore — Vite asset URL suffix, not a real module
import wasmUrl from "../wasm/pf.wasm?url";

export interface Entity {
  entity_group: string;
  start: number; // byte offset into original text
  end: number;
  score: number;
  text: string;
}

const MODEL_PATH = "/model.gguf";
const CACHE_NAME = "privacy-filter-v1";

type InMessage =
  | { type: "load"; url: string }
  | { type: "classify"; text: string; threshold: number };

type OutMessage =
  | { type: "ready" }
  | { type: "progress"; received: number; total: number; cached: boolean }
  | { type: "model-loaded"; loadMs: number; cached: boolean }
  | { type: "partial"; entities: Entity[]; chunkIndex: number; totalChunks: number }
  | { type: "result"; entities: Entity[]; classifyMs: number }
  | { type: "error"; message: string };

const post = (msg: OutMessage) =>
  (self as unknown as Worker).postMessage(msg);

let Module: any = null;

// ─── Initialize WASM on worker boot ───────────────────────────────────────────
async function boot() {
  try {
    Module = await PfModuleFactory({ locateFile: () => wasmUrl });
    post({ type: "ready" });
  } catch (err) {
    post({ type: "error", message: `WASM init: ${String(err)}` });
  }
}

// ─── Fetch model (Cache API, keyed by URL) ─────────────────────────────────────
async function fetchModel(url: string): Promise<{ buf: Uint8Array; cached: boolean }> {
  const cache = await caches.open(CACHE_NAME);
  const hit = await cache.match(url);
  if (hit) {
    return { buf: await readBody(hit, true), cached: true };
  }

  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);

  const clone = res.clone();
  const buf = await readBody(res, false);
  await cache.put(url, clone);
  return { buf, cached: false };
}

async function readBody(res: Response, cached: boolean): Promise<Uint8Array> {
  const total = Number(res.headers.get("content-length") ?? 0);
  const reader = res.body!.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    post({ type: "progress", received, total, cached });
  }

  const buf = new Uint8Array(received);
  let pos = 0;
  for (const c of chunks) { buf.set(c, pos); pos += c.length; }
  return buf;
}

// ─── Load ──────────────────────────────────────────────────────────────────────
async function loadModel(url: string) {
  const t0 = performance.now();
  const { buf, cached } = await fetchModel(url);
  Module.FS.writeFile(MODEL_PATH, buf);

  const load = Module.cwrap("pf_web_load", "number", ["string"]);
  const err = load(MODEL_PATH);
  if (err) {
    const getErr = Module.cwrap("pf_web_error", "string", []);
    throw new Error(getErr());
  }

  const loadMs = performance.now() - t0;
  post({ type: "model-loaded", loadMs, cached });
}

// ─── Text chunking ─────────────────────────────────────────────────────────────
// Split text into chunks at sentence boundaries using Intl.Segmenter, then
// batch sentences up to ~1500 chars so we don't pay per-chunk ggml overhead
// for tiny fragments. Intl.Segmenter gives us exact char indices, which we
// convert to UTF-8 byte offsets once via TextEncoder on the prefix.
interface Chunk {
  text: string;
  byteOffset: number; // UTF-8 byte offset into the original text
}

function chunkText(text: string): Chunk[] {
  const enc = new TextEncoder();
  const segmenter = new Intl.Segmenter("en", { granularity: "sentence" });

  const segments = [...segmenter.segment(text)];
  if (segments.length === 0) return [{ text, byteOffset: 0 }];

  const chunks: Chunk[] = [];
  let buf = "";
  let bufStart = 0; // char index in `text`

  for (const seg of segments) {
    const segText = seg.segment;
    const segStart = seg.index; // char index

    // Merge tiny segments (<120 chars) or grow the buffer.
    if (!buf) bufStart = segStart;
    buf += segText;

    // Flush when the buffer reaches ~200 chars, or when a long pause
    // (double newline) appears inside the segment.
    if (buf.length >= 200 || /\n\n/.test(segText)) {
      chunks.push({
        text: buf,
        byteOffset: enc.encode(text.slice(0, bufStart)).length,
      });
      buf = "";
    }
  }
  if (buf) {
    chunks.push({
      text: buf,
      byteOffset: enc.encode(text.slice(0, bufStart)).length,
    });
  }

  return chunks;
}

// ─── Streaming classify ───────────────────────────────────────────────────────
async function classifyStream(text: string, threshold: number) {
  const t0 = performance.now();
  const fn = Module.cwrap("pf_web_classify", "string", ["string", "number"]);

  const chunks = chunkText(text);
  const allEntities: Entity[] = [];

  for (let i = 0; i < chunks.length; i++) {
    const { text: chunkText, byteOffset } = chunks[i];
    const json = fn(chunkText, threshold);
    const data = JSON.parse(json);

    if (data.error) throw new Error(data.error);

    // Adjust byte offsets from chunk-local to document-global.
    for (const e of data as Entity[]) {
      allEntities.push({
        ...e,
        start: e.start + byteOffset,
        end: e.end + byteOffset,
      });
    }

    // Yield to the event loop so the worker can post the message.
    post({
      type: "partial",
      entities: [...allEntities],
      chunkIndex: i + 1,
      totalChunks: chunks.length,
    });

    // Let the main thread paint before processing the next chunk.
    await new Promise((r) => setTimeout(r, 0));
  }

  const classifyMs = performance.now() - t0;
  post({ type: "result", entities: allEntities, classifyMs });
}

// ─── Message router ────────────────────────────────────────────────────────────
self.addEventListener("message", async (e: MessageEvent<InMessage>) => {
  const msg = e.data;
  try {
    if (msg.type === "load") {
      await loadModel(msg.url);
    } else if (msg.type === "classify") {
      await classifyStream(msg.text, msg.threshold);
    }
  } catch (err) {
    post({ type: "error", message: String(err) });
  }
});

boot();
