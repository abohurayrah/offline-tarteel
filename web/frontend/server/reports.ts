import { Hono } from "hono";
import { randomUUID } from "node:crypto";
import { S3Client, PutObjectCommand, ListObjectsV2Command, GetObjectCommand } from "@aws-sdk/client-s3";

// R2 configuration (S3-compatible)
const R2_ACCOUNT_ID = "f82eb5e980c7982a05f36a8064fc9e7a";
const R2_ACCESS_KEY = process.env.R2_ACCESS_KEY || "a3207691cdbfae9d93bc5af6eada5fa2";
const R2_SECRET_KEY = process.env.R2_SECRET_KEY || "76de82afed70929c06ead379eea909a9f5f11188ef47393a65e5b9095519c4d3";
const R2_BUCKET = "itqan-feedback";
const R2_ENDPOINT = `https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com`;

const s3 = new S3Client({
  region: "auto",
  endpoint: R2_ENDPOINT,
  credentials: {
    accessKeyId: R2_ACCESS_KEY,
    secretAccessKey: R2_SECRET_KEY,
  },
});

/** Validate that an ID is a safe UUID (no path traversal) */
function isValidId(id: string): boolean {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(id);
}

export const reportsApp = new Hono();

// POST /api/reports — accept audio + metadata, upload to R2
reportsApp.post("/", async (c) => {
  const form = await c.req.formData();
  const audio = form.get("audio") as File | null;
  const metaRaw = form.get("metadata") as string | null;

  if (!audio || !metaRaw) {
    return c.json({ error: "Missing audio or metadata" }, 400);
  }

  let meta: Record<string, unknown>;
  try {
    meta = JSON.parse(metaRaw);
  } catch {
    return c.json({ error: "Invalid metadata JSON" }, 400);
  }

  const id = randomUUID();
  const audioBuffer = Buffer.from(await audio.arrayBuffer());

  const fullMeta = {
    id,
    ...meta,
    timestamp: new Date().toISOString(),
    userAgent: c.req.header("user-agent") || "",
    audioSizeBytes: audioBuffer.length,
  };

  try {
    // Upload audio to R2
    await s3.send(new PutObjectCommand({
      Bucket: R2_BUCKET,
      Key: `reports/${id}/audio.wav`,
      Body: audioBuffer,
      ContentType: "audio/wav",
    }));

    // Upload metadata to R2
    await s3.send(new PutObjectCommand({
      Bucket: R2_BUCKET,
      Key: `reports/${id}/meta.json`,
      Body: JSON.stringify(fullMeta, null, 2),
      ContentType: "application/json",
    }));

    return c.json({ id, status: "saved" }, 201);
  } catch (err) {
    console.error("R2 upload failed:", err);
    return c.json({ error: "Storage error" }, 500);
  }
});

// GET /api/reports — list all reports from R2
reportsApp.get("/", async (c) => {
  try {
    const list = await s3.send(new ListObjectsV2Command({
      Bucket: R2_BUCKET,
      Prefix: "reports/",
      Delimiter: "/",
    }));

    const reports = [];
    for (const prefix of list.CommonPrefixes || []) {
      try {
        const metaKey = `${prefix.Prefix}meta.json`;
        const obj = await s3.send(new GetObjectCommand({
          Bucket: R2_BUCKET,
          Key: metaKey,
        }));
        const body = await obj.Body?.transformToString();
        if (body) reports.push(JSON.parse(body));
      } catch { /* skip */ }
    }

    reports.sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
    );
    return c.json(reports);
  } catch {
    return c.json([]);
  }
});

// GET /api/reports/:id/audio — stream audio from R2
reportsApp.get("/:id/audio", async (c) => {
  const id = c.req.param("id");
  if (!isValidId(id)) return c.json({ error: "Invalid ID" }, 400);

  try {
    const obj = await s3.send(new GetObjectCommand({
      Bucket: R2_BUCKET,
      Key: `reports/${id}/audio.wav`,
    }));

    const body = await obj.Body?.transformToByteArray();
    if (!body) return c.json({ error: "Not found" }, 404);

    return new Response(body, {
      headers: {
        "Content-Type": "audio/wav",
        "Content-Length": String(body.length),
      },
    });
  } catch {
    return c.json({ error: "Not found" }, 404);
  }
});
