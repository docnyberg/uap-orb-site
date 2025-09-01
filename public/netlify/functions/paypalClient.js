// netlify/functions/paypalClient.js
const fetch = (...args) => import('node-fetch').then(({ default: f }) => f(...args));

const PAYPAL_BASE =
  process.env.NODE_ENV === 'production'
    ? 'https://api-m.paypal.com'
    : 'https://api-m.sandbox.paypal.com'; // use sandbox during testing

async function getAccessToken() {
  const client = process.env.PAYPAL_CLIENT_ID;
  const secret = process.env.PAYPAL_CLIENT_SECRET;

  const basic = Buffer.from(`${client}:${secret}`).toString('base64');
  const res = await fetch(`${PAYPAL_BASE}/v1/oauth2/token`, {
    method: 'POST',
    headers: {
      Authorization: `Basic ${basic}`,
      'Content-Type': 'application/x-www-form-urlencoded'
    },
    body: 'grant_type=client_credentials'
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`PayPal token error: ${res.status} ${text}`);
  }
  const json = await res.json();
  return json.access_token;
}

module.exports = {
  PAYPAL_BASE,
  getAccessToken
};
