import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";
import path from "path";

export default defineConfig({
  plugins: [
    VitePWA({
      registerType: "autoUpdate",
      workbox: {
        // Pre-cache app shell assets (JS, CSS, HTML, fonts)
        globPatterns: ["**/*.{js,css,html,woff2,woff,ttf}"],
        // Exclude the large ONNX model from precache (handled by runtimeCaching)
        globIgnores: ["**/*.onnx"],
        // Increase precache size limit for wasm files
        maximumFileSizeToCacheInBytes: 15 * 1024 * 1024,
        runtimeCaching: [
          {
            // ONNX model: cache on first load, serve from cache after
            urlPattern: /\.onnx$/,
            handler: "CacheFirst",
            options: {
              cacheName: "model-cache",
              expiration: { maxEntries: 2 },
              cacheableResponse: { statuses: [0, 200] },
              // Allow large model file (131MB)
              rangeRequests: false,
            },
          },
          {
            // Data JSON files (quran.json, vocab.json, mushaf-pages.json, etc.)
            urlPattern: /\.json$/,
            handler: "CacheFirst",
            options: {
              cacheName: "data-cache",
              expiration: { maxEntries: 20, maxAgeSeconds: 30 * 24 * 60 * 60 },
              cacheableResponse: { statuses: [0, 200] },
            },
          },
          {
            // WASM files
            urlPattern: /\.wasm$/,
            handler: "CacheFirst",
            options: {
              cacheName: "wasm-cache",
              expiration: { maxEntries: 10 },
              cacheableResponse: { statuses: [0, 200] },
            },
          },
          {
            // Font files
            urlPattern: /\.(?:woff2?|ttf|otf)$/,
            handler: "CacheFirst",
            options: {
              cacheName: "font-cache",
              expiration: {
                maxEntries: 20,
                maxAgeSeconds: 365 * 24 * 60 * 60,
              },
              cacheableResponse: { statuses: [0, 200] },
            },
          },
        ],
      },
      manifest: {
        name: "Tarteel - Quran Recognition",
        short_name: "Tarteel",
        description:
          "On-device Quran verse recognition. Recite and see your verses highlighted in real time.",
        start_url: "/",
        display: "standalone",
        background_color: "#FAFAFA",
        theme_color: "#C2A05B",
        icons: [
          {
            src: "/icon-192.png",
            sizes: "192x192",
            type: "image/png",
          },
          {
            src: "/icon-512.png",
            sizes: "512x512",
            type: "image/png",
          },
          {
            src: "/icon-512.png",
            sizes: "512x512",
            type: "image/png",
            purpose: "maskable",
          },
        ],
      },
    }),
  ],
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
