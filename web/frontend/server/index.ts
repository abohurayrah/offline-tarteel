import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import { Hono } from "hono";
import { adminApp } from "./admin.js";
import { reportsApp } from "./reports.js";

const app = new Hono();

// COOP/COEP for SharedArrayBuffer + cache control
app.use("*", async (c, next) => {
  await next();
  c.header("Cross-Origin-Opener-Policy", "same-origin");
  c.header("Cross-Origin-Embedder-Policy", "require-corp");

  const path = new URL(c.req.url).pathname;
  if (path === "/sw.js" || path === "/index.html" || path === "/") {
    c.header("Cache-Control", "no-cache, no-store, must-revalidate");
  } else if (path.startsWith("/assets/")) {
    c.header("Cache-Control", "public, max-age=31536000, immutable");
  }
});

// API routes
app.get("/api/health", (c) => c.json({ ok: true }));

// Mount reports API
app.route("/api/reports", reportsApp);

// Mount admin dashboard
app.route("/admin", adminApp);

// Serve static files from Vite build
app.use("/*", serveStatic({ root: "./dist" }));

// SPA fallback
app.get("/*", serveStatic({ root: "./dist", path: "index.html" }));

const port = parseInt(process.env.PORT || "5000");
console.log(`Server running on port ${port}`);
serve({ fetch: app.fetch, port });
