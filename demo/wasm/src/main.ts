import type { Entity } from "./worker";

const DEFAULT_URL =
  "https://huggingface.co/LocalAI-io/privacy-filter-GGUF/resolve/main/privacy-filter-q8.gguf";
const STORAGE_KEY = "pf-model-url";

// ─── Worker message types ────────────────────────────────────────────────────
type OutMessage =
  | { type: "load"; url: string }
  | { type: "classify"; text: string; threshold: number };

let worker: Worker | null = null;
let modelLoaded = false;
let loadedUrl = ""; // URL of the model currently in ggml

// ─── DOM ─────────────────────────────────────────────────────────────────────
const $ = <T extends HTMLElement = HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

const input = $<HTMLTextAreaElement>("input");
const output = $("output");
const stats = $("stats");
const entityCount = $("entity-count");
const table = $<HTMLTableElement>("table");
const tableEmpty = $("table-empty");
const scanBtn = $<HTMLButtonElement>("scan");
const loadBtn = $<HTMLButtonElement>("load-btn");
const modelUrl = $<HTMLInputElement>("model-url");
const modelStatus = $("model-status");
const modelInfo = $("model-info");
const thr = $<HTMLInputElement>("thr");
const thrVal = $("thr-val");
const progressEl = $("progress");
const progressBar = $<HTMLDivElement>("progress-bar");
const progressContainer = $<HTMLDivElement>("progress-container");
const scanProgress = $("scan-progress");
const scanProgressText = $("scan-progress-text");
const scanProgressFill = $<HTMLDivElement>("scan-progress-fill");

function setModelStatus(state: "idle" | "loading" | "loaded" | "stale" | "error", text: string) {
  modelStatus.className = `model-status ${state}`;
  modelStatus.textContent = text;
}

function esc(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// Byte-offset safe highlighting — the C engine returns UTF-8 byte offsets.
function highlight(text: string, entities: Entity[]): string {
  if (entities.length === 0) return esc(text);
  const enc = new TextEncoder();
  const dec = new TextDecoder();
  const bytes = enc.encode(text);
  const sorted = [...entities].sort((a, b) => a.start - b.start);
  let html = "";
  let last = 0;
  for (const e of sorted) {
    if (e.start < last) continue;
    html += esc(dec.decode(bytes.subarray(last, e.start)));
    const span = esc(dec.decode(bytes.subarray(e.start, e.end)));
    html += `<mark class="pii" data-cat="${e.entity_group} (${e.score.toFixed(2)})">${span}</mark>`;
    last = e.end;
  }
  html += esc(dec.decode(bytes.subarray(last)));
  return html;
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function renderTable(entities: Entity[]) {
  const tb = table.querySelector("tbody")!;
  tb.innerHTML = entities
    .map(
      (e, i) =>
        `<tr><td>${i + 1}</td><td><span class="pill">${e.entity_group}</span></td><td>${esc(e.text)}</td><td>${e.score.toFixed(3)}</td></tr>`,
    )
    .join("");
  table.style.display = entities.length ? "" : "none";
  tableEmpty.style.display = entities.length ? "none" : "";
}

function showScanProgress(text: string, pct: number) {
  scanProgress.classList.add("visible");
  scanProgressText.textContent = text;
  scanProgressFill.style.width = `${pct}%`;
}

function hideScanProgress() {
  scanProgress.classList.remove("visible");
}

// ─── Worker message handler ───────────────────────────────────────────────────
function onMessage(e: MessageEvent) {
  const msg = e.data;
  switch (msg.type) {
    case "ready":
      loadBtn.disabled = false;
      loadBtn.textContent = "Load model";
      setModelStatus("idle", "not loaded");
      break;

    case "progress": {
      const { received, total, cached } = msg;
      progressContainer.style.display = "";
      const pct = total ? ((received / total) * 100).toFixed(1) : "?";
      const tag = cached ? " (cached)" : "";
      progressEl.textContent =
        `${(received / 1e6).toFixed(0)} / ${total ? (total / 1e6).toFixed(0) : "?"} MB (${pct}%)${tag}`;
      progressBar.style.width = cached ? "100%" : `${pct}%`;
      break;
    }

    case "model-loaded":
      modelLoaded = true;
      loadedUrl = modelUrl.value.trim();
      scanBtn.disabled = false;
      input.disabled = false;
      progressContainer.style.display = "none";
      progressEl.textContent = "";
      setModelStatus("loaded", "loaded");
      modelInfo.textContent =
        `${formatMs(msg.loadMs)}${msg.cached ? " · cached" : ""}`;
      loadBtn.disabled = false;
      loadBtn.textContent = "Reload";
      stats.textContent = "Ready";
      break;

    case "partial": {
      const { entities, chunkIndex, totalChunks } = msg;
      const text = input.value;
      output.innerHTML = highlight(text, entities);
      renderTable(entities);
      const pct = Math.round((chunkIndex / totalChunks) * 100);
      showScanProgress(
        `Scanning chunk ${chunkIndex}/${totalChunks} — ${entities.length} entit${entities.length === 1 ? "y" : "ies"}`,
        pct,
      );
      entityCount.textContent = String(entities.length);
      break;
    }

    case "result": {
      const { entities, classifyMs } = msg;
      const text = input.value;
      output.innerHTML = highlight(text, entities);
      renderTable(entities);
      showScanProgress(
        entities.length === 0
          ? `No entities — ${formatMs(classifyMs)}`
          : `${entities.length} entit${entities.length === 1 ? "y" : "ies"} in ${formatMs(classifyMs)}`,
        100,
      );
      stats.textContent =
        entities.length === 0
          ? `No entities (${formatMs(classifyMs)})`
          : `${entities.length} entit${entities.length === 1 ? "y" : "ies"} in ${formatMs(classifyMs)}`;
      entityCount.textContent = String(entities.length);
      scanBtn.disabled = false;
      scanBtn.textContent = "Scan";
      break;
    }

    case "error":
      showScanProgress(`Error: ${msg.message}`, 100);
      progressEl.innerHTML = `<span style="color:#f85149">${esc(msg.message)}</span>`;
      setModelStatus("error", "error");
      stats.textContent = "Error";
      scanBtn.disabled = false;
      scanBtn.textContent = "Scan";
      loadBtn.disabled = false;
      loadBtn.textContent = "Retry";
      break;
  }
}

// ─── Actions ──────────────────────────────────────────────────────────────────
function loadModel() {
  const url = modelUrl.value.trim();
  if (!url) return;
  localStorage.setItem(STORAGE_KEY, url);
  loadBtn.disabled = true;
  loadBtn.textContent = "Loading…";
  setModelStatus("loading", "loading");
  modelInfo.textContent = "";
  progressContainer.style.display = "";
  progressBar.style.width = "0";
  progressEl.textContent = "";
  // Disable scan while loading a new model
  scanBtn.disabled = true;
  modelLoaded = false;
  const msg: OutMessage = { type: "load", url };
  worker!.postMessage(msg);
}

// If the URL input changes after a model is loaded, mark it stale so users
// know they need to reload — the running ggml context still holds the old one.
function checkStale() {
  if (!modelLoaded) return;
  if (modelUrl.value.trim() !== loadedUrl) {
    setModelStatus("stale", "reload needed");
    loadBtn.textContent = "Reload";
    modelInfo.textContent = "URL changed";
  } else {
    setModelStatus("loaded", "loaded");
    modelInfo.textContent = "";
  }
}

function classify() {
  const text = input.value;
  if (!text || !modelLoaded) return;
  scanBtn.disabled = true;
  scanBtn.textContent = "Scanning…";
  output.innerHTML = '<span style="color:var(--muted)">Scanning…</span>';
  entityCount.textContent = "";
  renderTable([]);
  showScanProgress("Starting…", 0);
  const msg: OutMessage = {
    type: "classify",
    text,
    threshold: parseFloat(thr.value),
  };
  worker!.postMessage(msg);
}

// ─── Boot ──────────────────────────────────────────────────────────────────────
modelUrl.value = localStorage.getItem(STORAGE_KEY) || DEFAULT_URL;

worker = new Worker(new URL("./worker.ts", import.meta.url), {
  type: "module",
});
worker.addEventListener("message", onMessage);

loadBtn.addEventListener("click", loadModel);
modelUrl.addEventListener("input", checkStale);
scanBtn.addEventListener("click", classify);
thr.addEventListener("input", () => {
  thrVal.textContent = parseFloat(thr.value).toFixed(2);
});
