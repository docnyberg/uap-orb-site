// netlify/functions/get-r2-url.js
import {
  S3Client,
  GetObjectCommand,
  HeadObjectCommand,
  ListObjectsV2Command
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const s3 = new S3Client({
  region: "auto",
  endpoint: `https://${process.env.CF_ACCOUNT_ID}.r2.cloudflarestorage.com`,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY,
  },
  forcePathStyle: true,
});

function ensureMp4(name) {
  if (!name) return null;
  let v = String(name).trim().split(/[\\/]/).pop();
  v = v.replace(/(\.mp4)+$/i, ""); // collapse .mp4.mp4
  return v + ".mp4";
}

function pickBest(candidates, base) {
  // prefer the shortest extra suffix after base (handles stray digits, etc.)
  const scored = candidates
    .map(k => {
      const noExt = k.replace(/\.mp4$/i, "");
      const extraLen = Math.max(0, noExt.length - base.length);
      return { k, extraLen };
    })
    .sort((a, b) => a.extraLen - b.extraLen);
  return scored[0]?.k;
}

export async function handler(event) {
  try {
    const raw = (event.queryStringParameters?.file || "").trim();
    if (!/^[A-Za-z0-9._/-]+\.mp4$/i.test(raw)) {
      return { statusCode: 400, body: "Bad file name" };
    }

    const Bucket = process.env.R2_BUCKET;
    let Key = ensureMp4(raw);
    const expiresIn = Number(process.env.R2_DOWNLOAD_TTL || 900);

    // 1) exact key?
    try {
      await s3.send(new HeadObjectCommand({ Bucket, Key }));
    } catch {
      // 2) fallback by prefix
      const base = Key.replace(/\.mp4$/i, "");
      const listed = await s3.send(
        new ListObjectsV2Command({ Bucket, Prefix: base })
      );
      const candidates = (listed.Contents || [])
        .map(o => o.Key)
        .filter(k => k && k.toLowerCase().endsWith(".mp4"));
      if (!candidates.length) {
        return { statusCode: 404, body: "Not found in R2" };
      }
      Key = pickBest(candidates, base);
    }

    const cmd = new GetObjectCommand({
      Bucket,
      Key,
      ResponseContentDisposition: `attachment; filename="${Key.split("/").pop()}"`
    });
    const url = await getSignedUrl(s3, cmd, { expiresIn });

    return {
      statusCode: 200,
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ url, ttl: expiresIn })
    };
  } catch (err) {
    const code = err?.$metadata?.httpStatusCode || 500;
    return { statusCode: code, body: err?.message || "Error generating URL" };
  }
}
