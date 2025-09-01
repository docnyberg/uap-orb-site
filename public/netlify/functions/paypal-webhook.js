// netlify/functions/paypal-webhook.js
const fetch = (...args) => import('node-fetch').then(({ default: f }) => f(...args));
const { PAYPAL_BASE, getAccessToken } = require('./paypalClient');

const WEBHOOK_ID = process.env.PAYPAL_WEBHOOK_ID; // set in Netlify env

/**
 * PayPal sends these headers; names are case-insensitive but Netlify normalizes to lowercase.
 * We'll read the lowercase keys from event.headers.
 */
function pickPaypalHeaders(headers = {}) {
  return {
    transmission_id: headers['paypal-transmission-id'],
    transmission_time: headers['paypal-transmission-time'],
    cert_url: headers['paypal-cert-url'],
    auth_algo: headers['paypal-auth-algo'],
    transmission_sig: headers['paypal-transmission-sig']
  };
}

exports.handler = async (event) => {
  try {
    if (event.httpMethod !== 'POST') {
      return { statusCode: 405, body: 'Method Not Allowed' };
    }
    if (!WEBHOOK_ID) {
      console.error('PAYPAL_WEBHOOK_ID missing');
      return { statusCode: 500, body: 'Webhook not configured' };
    }

    const rawBody = event.body || ''; // raw JSON string from PayPal
    const headers = pickPaypalHeaders(event.headers || {});
    const webhookEvent = JSON.parse(rawBody || '{}');

    // 1) Verify signature with PayPal
    const token = await getAccessToken();
    const verifyRes = await fetch(`${PAYPAL_BASE}/v1/notifications/verify-webhook-signature`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        transmission_id: headers.transmission_id,
        transmission_time: headers.transmission_time,
        cert_url: headers.cert_url,
        auth_algo: headers.auth_algo,
        transmission_sig: headers.transmission_sig,
        webhook_id: WEBHOOK_ID,
        webhook_event: webhookEvent
      })
    });

    if (!verifyRes.ok) {
      const text = await verifyRes.text().catch(() => '');
      console.error('PayPal verify error:', verifyRes.status, text);
      return { statusCode: 400, body: 'Signature verify failed' };
    }

    const verifyJson = await verifyRes.json();
    if (verifyJson.verification_status !== 'SUCCESS') {
      console.warn('PayPal webhook verification_status:', verifyJson.verification_status);
      return { statusCode: 400, body: 'Invalid signature' };
    }

    // 2) At this point the event is authentic; act on it if you want
    // Examples (commonly interesting events):
    // - PAYMENT.CAPTURE.COMPLETED
    // - CHECKOUT.ORDER.APPROVED
    // - CHECKOUT.ORDER.COMPLETED
    const eventType = webhookEvent.event_type;
    const resource = webhookEvent.resource || {};

    // Minimal example: log a few fields (Netlify logs)
    console.log('[PayPal webhook verified]', {
      eventType,
      id: resource.id,
      amount: resource.amount || resource.purchase_units?.[0]?.amount,
      status: resource.status || webhookEvent.summary
    });

    // Optional: you could store this in a database, reconcile orders, etc.
    // If you plan to grant tokens via webhook, you'd lookup buyer -> issue JWT here.

    // 3) Always return 200 fast so PayPal stops retrying
    return { statusCode: 200, body: 'OK' };
  } catch (err) {
    console.error('Webhook handler error:', err);
    return { statusCode: 500, body: 'Server error' };
  }
};
