import { defineConfig } from "vite";
import { cpSync, existsSync } from "node:fs";
import { resolve } from "node:path";

// Copy build artifacts (Emscripten pf.js/pf.wasm + Cloudflare _headers) into
// the dist/ root after Vite finishes the bundle. The pf.* files are generated
// by build.sh (Emscripten) and live outside Vite's module graph.
function copyStatic() {
  return {
    name: "copy-static",
    closeBundle() {
      const copies: Array<[string, string]> = [
        ["public/_headers", "dist/_headers"],
        ["wasm/pf.js", "dist/pf.js"],
        ["wasm/pf.wasm", "dist/pf.wasm"],
      ];
      for (const [src, dst] of copies) {
        const s = resolve(process.cwd(), src);
        if (existsSync(s)) {
          cpSync(s, resolve(process.cwd(), dst));
          console.log(`  copied ${src} → ${dst}`);
        } else {
          console.warn(`  ! missing ${src} — run ./build.sh first`);
        }
      }
    },
  };
}

export default defineConfig({
  server: { port: 5174 },
  build: {
    target: "esnext",
    rollupOptions: {
      external: ["/pf.js", "/pf.wasm", "/pf.worker.js"],
    },
  },
  plugins: [copyStatic()],
});
