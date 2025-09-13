// netlify/functions/get-r2-url.js
import { S3Client, GetObjectCommand } from "@aws-sdk/client-s3";
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

export async function handler(event) {
  try {
    const file = (event.queryStringParameters?.file || "").trim();
    if (!/^[A-Za-z0-9._-]+\.mp4$/.test(file)) {
      return { statusCode: 400, body: "Bad file name" };
    }
    const Bucket = process.env.R2_BUCKET;
    const Key = file;
    const expiresIn = Number(process.env.R2_DOWNLOAD_TTL || 600);

    const cmd = new GetObjectCommand({
      Bucket,
      Key,
      ResponseContentDisposition: `attachment; filename="${file}"`,
    });

    const url = await getSignedUrl(s3, cmd, { expiresIn });
    return {
      statusCode: 200,
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ url, ttl: expiresIn }),
    };
  } catch (err) {
    const code = err?.$metadata?.httpStatusCode || 500;
    return { statusCode: code, body: err?.message || "Error generating URL" };
  }
}
