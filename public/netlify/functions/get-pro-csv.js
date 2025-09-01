// public/netlify/functions/get-pro-csv.js
const fs = require('fs');
const path = require('path');
const jwt = require('jsonwebtoken');

const JWT_SECRET = process.env.PRO_JWT_SECRET || 'replace_me_secret';

/**
 * GET /get-pro-csv with Authorization: Bearer <token>
 * Returns the CSV only if token is valid and tier === 'pro'
 */
exports.handler = async (event) => {
  try {
    const auth = event.headers.authorization || '';
    const m = auth.match(/^Bearer\s+(.+)$/i);
    if (!m) return { statusCode: 401, body: 'Missing token' };

    const token = m[1];
    let payload;
    try {
      payload = jwt.verify(token, JWT_SECRET);
      if (payload.tier !== 'pro') throw new Error('not pro');
    } catch (e) {
      return { statusCode: 403, body: 'Invalid token' };
    }

    const filePath = path.join(process.cwd(), 'public', 'atlas_pro.csv');
    if (!fs.existsSync(filePath)) {
      return { statusCode: 404, body: 'atlas_pro.csv not found' };
    }

    const csv = fs.readFileSync(filePath, 'utf8');
    return {
      statusCode: 200,
      headers: { 'Content-Type': 'text/csv' },
      body: csv
    };
  } catch (err) {
    console.error('get-pro-csv error:', err);
    return { statusCode: 500, body: 'Server error' };
  }
};
