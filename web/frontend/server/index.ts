import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import { Hono } from "hono";
import { adminApp } from "./admin.js";
import { diagnosticsApp } from "./diagnostics.js";
import { reportsApp } from "./reports.js";

const app = new Hono();

// COOP/COEP for SharedArrayBuffer + cache control + Content-Type fixes
app.use("*", async (c, next) => {
  await next();
  c.header("Cross-Origin-Opener-Policy", "same-origin");
  c.header("Cross-Origin-Embedder-Policy", "require-corp");

  const path = new URL(c.req.url).pathname;

  // Correct Content-Type for WASM files
  if (path.endsWith(".wasm")) {
    c.header("Content-Type", "application/wasm");
  }
  // Correct Content-Type for ONNX model files
  if (path.endsWith(".onnx")) {
    c.header("Content-Type", "application/octet-stream");
  }

  // Cache control: no-cache for HTML/SW, immutable for hashed assets, long-cache for models/data
  if (path === "/sw.js" || path === "/index.html" || path === "/") {
    c.header("Cache-Control", "no-cache, no-store, must-revalidate");
  } else if (path.startsWith("/assets/")) {
    c.header("Cache-Control", "public, max-age=31536000, immutable");
  } else if (path.endsWith(".onnx")) {
    c.header("Cache-Control", "public, max-age=604800"); // 7 days for model
  } else if (path.endsWith(".json") && !path.startsWith("/api/")) {
    c.header("Cache-Control", "public, max-age=86400"); // 1 day for data JSON
  } else if (path.endsWith(".wasm")) {
    c.header("Cache-Control", "public, max-age=604800"); // 7 days for WASM
  } else if (path.endsWith(".woff2") || path.endsWith(".woff") || path.endsWith(".ttf")) {
    c.header("Cache-Control", "public, max-age=31536000, immutable");
  }
});

// API routes
app.get("/api/health", (c) => c.json({ ok: true }));

// Cache-bust endpoint to clear stale service workers
app.get("/api/clear-cache", (c) => {
  c.header("Clear-Site-Data", '"cache", "storage"');
  return c.json({ cleared: true });
});

// Mount reports API
app.route("/api/reports", reportsApp);
app.route("/api/diagnostics", diagnosticsApp);

// Mount admin dashboard
app.route("/admin", adminApp);

// Serve static files from Vite build
app.use("/*", serveStatic({ root: "./dist" }));

// SPA fallback
app.get("/*", serveStatic({ root: "./dist", path: "index.html" }));

const port = parseInt(process.env.PORT || "5000");
console.log(`Server running on port ${port}`);
serve({ fetch: app.fetch, port });
