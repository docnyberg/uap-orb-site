import { createServer } from 'http';
import { createReadStream, existsSync, statSync } from 'fs';
import { extname, join, resolve } from 'path';

const __dirname = resolve();
const rootDir = resolve(__dirname, 'public');
const port = Number(process.env.PORT || 8888);

const mimeTypes = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.csv': 'text/csv; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.mp4': 'video/mp4',
  '.txt': 'text/plain; charset=utf-8',
};

function sanitizePath(requestPath) {
  try {
    const decoded = decodeURIComponent(requestPath);
    const safePath = decoded.replace(/\0/g, '');
    const fullPath = resolve(rootDir, '.' + safePath);
    if (!fullPath.startsWith(rootDir)) {
      return null;
    }
    return fullPath;
  } catch {
    return null;
  }
}

async function handleRequest(req, res) {
  const url = new URL(req.url || '/', `http://${req.headers.host}`);
  let filePath = sanitizePath(url.pathname);

  if (!filePath) {
    res.statusCode = 400;
    res.end('Bad request');
    return;
  }

  try {
    const stats = statSync(filePath);
    if (stats.isDirectory()) {
      filePath = join(filePath, 'index.html');
    }
  } catch (err) {
    if (err.code === 'ENOENT') {
      // fallback to index.html for client-side routing
      filePath = join(rootDir, 'index.html');
    } else {
      res.statusCode = 500;
      res.end('Server error');
      return;
    }
  }

  if (!existsSync(filePath)) {
    res.statusCode = 404;
    res.end('Not found');
    return;
  }

  const ext = extname(filePath).toLowerCase();
  const contentType = mimeTypes[ext] || 'application/octet-stream';

  res.statusCode = 200;
  res.setHeader('Content-Type', contentType);

  const stream = createReadStream(filePath);
  stream.on('error', () => {
    res.statusCode = 500;
    res.end('Server error');
  });
  stream.pipe(res);
}

const server = createServer((req, res) => {
  handleRequest(req, res).catch((err) => {
    res.statusCode = 500;
    res.end('Server error');
  });
});

server.listen(port, () => {
  console.log(`Static server running at http://localhost:${port}`);
});

process.on('SIGINT', () => {
  server.close(() => process.exit(0));
});
process.on('SIGTERM', () => {
  server.close(() => process.exit(0));
});
