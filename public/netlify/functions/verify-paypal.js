// public/netlify/functions/verify-paypal.js
const jwt = require('jsonwebtoken');
const fetch = (...args) => import('node-fetch').then(({ default: f }) => f(...args));
const { PAYPAL_BASE, getAccessToken } = require('./paypalClient');

const PRICE = parseFloat(process.env.PRO_UPGRADE_PRICE || '1.99');
const CURRENCY = process.env.PRO_UPGRADE_CURRENCY || 'USD';
const JWT_SECRET = process.env.PRO_JWT_SECRET || 'replace_me_secret';

/**
 * Verify order via PayPal Orders API and issue a JWT if completed and amount matches.
 * POST body: { orderID: "..." }
 * Response: { ok: true, token: "..." } or error
 */
exports.handler = async (event) => {
  try {
    if (event.httpMethod !== 'POST') {
      return { statusCode: 405, body: 'Method Not Allowed' };
    }

    let body;
    try { body = JSON.parse(event.body || '{}'); } catch { body = {}; }
    const orderID = body.orderID;
    if (!orderID) {
      return { statusCode: 400, body: 'Missing orderID' };
    }

    // Get PayPal REST token
    const accessToken = await getAccessToken();
    // Retrieve order info
    const res = await fetch(`${PAYPAL_BASE}/v2/checkout/orders/${orderID}`, {
      headers: { Authorization: `Bearer ${accessToken}` }
    });

    if (!res.ok) {
      const text = await res.text().catch(() => '');
      console.error('PayPal /orders error:', res.status, text);
      return { statusCode: 400, body: 'Unable to verify order' };
    }

    const order = await res.json();
    // Expect COMPLETED (captured). If you’re using onApprove+capture client-side, you may need to call capture or rely on this status.
    if (order.status !== 'COMPLETED') {
      return { statusCode: 402, body: 'Payment not completed.' };
    }

    const unit = (order.purchase_units && order.purchase_units[0]) || {};
    const amt = parseFloat(unit.amount?.value || '0');
    const cur = unit.amount?.currency_code || '';

    if (amt !== PRICE || cur !== CURRENCY) {
      console.warn('Amount/currency mismatch:', { amt, cur, expected: PRICE, CURRENCY });
      return { statusCode: 400, body: 'Amount/currency mismatch.' };
    }

    // Issue a JWT for Pro unlock
    const token = jwt.sign(
      { tier: 'pro', amount: amt, currency: cur, iat: Math.floor(Date.now()/1000) },
      JWT_SECRET,
      { expiresIn: '180 days' }
    );

    return {
      statusCode: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ok: true, token })
    };
  } catch (err) {
    console.error('verify-paypal error:', err);
    return { statusCode: 500, body: 'Server error' };
  }
};
