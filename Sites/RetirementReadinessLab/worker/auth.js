// Firebase issues the Google identity token; this Worker verifies it before billing.
export const OWNER_CHATGPT_USER_ID = '028696a7-7846-4822-a4c1-67026aa2383f';
export const OWNER_GOOGLE_EMAIL = 'rtsoliday@gmail.com';
export const OWNER_FIREBASE_UID = 'xYPnJEpGrHfTJmBtFUnXlAQSdzU2';
const JWKS_URL = 'https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com';
const encoder = new TextDecoder();
let cachedKeys = new Map();
let cacheUntil = 0;

function decodePart(part) {
  if (!/^[A-Za-z0-9_-]+$/.test(part)) throw new Error('Invalid identity token');
  const value = part.replace(/-/g, '+').replace(/_/g, '/');
  return Uint8Array.from(atob(value.padEnd(Math.ceil(value.length / 4) * 4, '=')), c => c.charCodeAt(0));
}
function parsePart(part) { return JSON.parse(encoder.decode(decodePart(part))); }
function firebaseConfigured(env) {
  return Boolean(env.FIREBASE_API_KEY && env.FIREBASE_AUTH_DOMAIN && env.FIREBASE_PROJECT_ID && env.FIREBASE_APP_ID);
}
async function firebaseKey(kid, env) {
  if (!env.FIREBASE_JWKS_FETCH && Date.now() < cacheUntil && cachedKeys.has(kid)) return cachedKeys.get(kid);
  const response = await (env.FIREBASE_JWKS_FETCH || fetch)(JWKS_URL, { signal: AbortSignal.timeout(10000) });
  if (!response.ok) throw new Error('Identity keys unavailable');
  const payload = await response.json();
  if (!Array.isArray(payload.keys)) throw new Error('Identity keys unavailable');
  const next = new Map();
  for (const jwk of payload.keys) {
    if (jwk.kty !== 'RSA' || (jwk.alg && jwk.alg !== 'RS256') || typeof jwk.kid !== 'string') continue;
    next.set(jwk.kid, await crypto.subtle.importKey('jwk', jwk, { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' }, false, ['verify']));
  }
  if (!env.FIREBASE_JWKS_FETCH) {
    cachedKeys = next;
    const maxAge = Number(response.headers.get('cache-control')?.match(/max-age=(\d+)/)?.[1] || 300);
    cacheUntil = Date.now() + Math.min(Math.max(maxAge, 60), 3600) * 1000;
  }
  const key = next.get(kid);
  if (!key) throw new Error('Unrecognized identity key');
  return key;
}
export async function verifyFirebaseToken(token, env) {
  if (!firebaseConfigured(env) || typeof token !== 'string' || token.length > 8192) throw new Error('Invalid identity token');
  const parts = token.split('.');
  if (parts.length !== 3) throw new Error('Invalid identity token');
  const header = parsePart(parts[0]);
  const claims = parsePart(parts[1]);
  if (header.alg !== 'RS256' || typeof header.kid !== 'string' || !header.kid || claims.aud !== env.FIREBASE_PROJECT_ID || claims.iss !== `https://securetoken.google.com/${env.FIREBASE_PROJECT_ID}`) throw new Error('Invalid identity token');
  const now = Math.floor(Date.now() / 1000);
  if (!Number.isInteger(claims.exp) || claims.exp <= now || !Number.isInteger(claims.iat) || claims.iat > now + 60 || !Number.isInteger(claims.auth_time) || claims.auth_time > now + 60 || !/^[A-Za-z0-9_-]{1,128}$/.test(claims.sub || '')) throw new Error('Invalid identity token');
  const provider = claims.firebase?.sign_in_provider;
  if (provider !== 'google.com') throw new Error('Unsupported identity provider');
  const key = await firebaseKey(header.kid, env);
  const valid = await crypto.subtle.verify('RSASSA-PKCS1-v1_5', key, decodePart(parts[2]), new TextEncoder().encode(`${parts[0]}.${parts[1]}`));
  if (!valid) throw new Error('Invalid identity token');
  return { kind: 'firebase', id: claims.sub, provider: 'google', verifiedEmail: claims.email_verified === true && typeof claims.email === 'string' ? claims.email.trim().toLowerCase() : null };
}
export function chatgptPrincipal(request) {
  const id = request.headers.get('oai-authenticated-user-id');
  return id ? { kind: 'chatgpt', id, provider: 'chatgpt' } : null;
}
export async function firebasePrincipal(request, env) {
  const authorization = request.headers.get('authorization');
  if (!authorization) return null;
  const match = /^Bearer ([A-Za-z0-9_.-]+)$/.exec(authorization);
  if (!match) throw new Error('Invalid identity token');
  return verifyFirebaseToken(match[1], env);
}
export async function principal(request, env) {
  return (await firebasePrincipal(request, env)) || chatgptPrincipal(request);
}
export function authConfig(request, env) {
  const enabled = firebaseConfigured(env);
  return new Response(JSON.stringify({
    configured: enabled,
    firebase: enabled ? { apiKey: env.FIREBASE_API_KEY, authDomain: env.FIREBASE_AUTH_DOMAIN, projectId: env.FIREBASE_PROJECT_ID, appId: env.FIREBASE_APP_ID } : null,
    chatgptSignedIn: Boolean(chatgptPrincipal(request)),
    providers: { google: enabled && env.FIREBASE_GOOGLE_ENABLED === 'true' },
  }), { headers: { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' } });
}
