import { defineConfig } from "vite";
import path from "path";

export default defineConfig({
  worker: {
    format: "es",
  },
  optimizeDeps: {
    exclude: ["@huggingface/transformers", "onnxruntime-web"],
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
    fs: {
      // Allow serving audio-samples from test/ during development
      allow: ["."],
    },
  },
  resolve: {
    alias: {
      // Allow benchmark to import from src/
      "@": path.resolve(__dirname, "src"),
    },
  },
});
