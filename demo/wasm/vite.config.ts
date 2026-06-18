import { defineConfig } from "vite";

export default defineConfig({
  server: { port: 5174 },
  build: { target: "esnext" },
  // Don't process the Emscripten output — treat it as a static asset
  assetsInclude: ["**/*.wasm"],
});
