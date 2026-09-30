// Firebase issues the Google identity token; this Worker verifies it before billing.
export const OWNER_CHATGPT_USER_ID = '028696a7-7846-4822-a4c1-67026aa2383f';
export const OWNER_GOOGLE_EMAIL = 'rtsoliday@gmail.com';
export const OWNER_FIREBASE_UID = 'xYPnJEpGrHfTJmBtFUnXlAQSdzU2';
const JWKS_URL = 'https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com';
const decoder = new TextDecoder();
// A fresh key set is refetched for an unknown key ID at most once a minute, so
// forged tokens cannot force a Google key download on every request.
const KEY_REFETCH_MS = 60000;

// Key-service failures cannot establish whether a token is valid or invalid.
// Keep them distinct from rejected credentials at the HTTP boundary.
export class IdentityKeysUnavailableError extends Error {
  constructor() { super('Identity verification is temporarily unavailable. Please retry.'); this.name = 'IdentityKeysUnavailableError'; }
}

function decodePart(part) {
  if (!/^[A-Za-z0-9_-]+$/.test(part)) throw new Error('Invalid identity token');
  const value = part.replace(/-/g, '+').replace(/_/g, '/');
  return Uint8Array.from(atob(value.padEnd(Math.ceil(value.length / 4) * 4, '=')), c => c.charCodeAt(0));
}
function parsePart(part) { return JSON.parse(decoder.decode(decodePart(part))); }
function firebaseConfigured(env) {
  return Boolean(env.FIREBASE_API_KEY && env.FIREBASE_AUTH_DOMAIN && env.FIREBASE_PROJECT_ID && env.FIREBASE_APP_ID);
}
async function loadIdentityKeys(fetcher) {
  try {
    const response = await fetcher(JWKS_URL, { signal: AbortSignal.timeout(10000) });
    if (!response.ok) throw new Error('Identity keys unavailable');
    const payload = await response.json();
    if (!Array.isArray(payload?.keys)) throw new Error('Invalid identity keys response');
    const keys = new Map();
    for (const jwk of payload.keys) {
      if (!jwk || jwk.kty !== 'RSA' || (jwk.alg && jwk.alg !== 'RS256') || typeof jwk.kid !== 'string' || !jwk.kid) continue;
      keys.set(jwk.kid, await crypto.subtle.importKey('jwk', jwk, { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' }, false, ['verify']));
    }
    if (!keys.size) throw new Error('No usable identity keys');
    const maxAge = Number(response.headers.get('cache-control')?.match(/max-age=(\d+)/)?.[1] || 300);
    return { keys, maxAgeSeconds: Math.min(Math.max(maxAge, 60), 3600) };
  } catch { throw new IdentityKeysUnavailableError(); }
}
// Returns a key lookup that caches the key set for its max-age. Concurrent
// refreshes share one download; failed downloads are retried on the next call.
export function identityKeyCache(fetcher, now = () => Date.now()) {
  let keys = new Map(), expires = 0, loadedAt = -Infinity, loading = null;
  return async kid => {
    const time = now();
    if (time < expires) {
      if (keys.has(kid)) return keys.get(kid);
      // A successfully loaded key set that excludes this token's key is a rejection.
      if (time - loadedAt < KEY_REFETCH_MS) throw new Error('Unrecognized identity key');
    }
    loading ||= loadIdentityKeys(fetcher).then(result => {
      keys = result.keys; loadedAt = now(); expires = loadedAt + result.maxAgeSeconds * 1000;
    }).finally(() => { loading = null; });
    await loading;
    const key = keys.get(kid);
    if (!key) throw new Error('Unrecognized identity key');
    return key;
  };
}
const cachedIdentityKey = identityKeyCache((url, init) => fetch(url, init));
async function firebaseKey(kid, env) {
  // An injected fetch models each key-service response, so it is never cached.
  if (env.FIREBASE_JWKS_FETCH) {
    const key = (await loadIdentityKeys(env.FIREBASE_JWKS_FETCH)).keys.get(kid);
    if (!key) throw new Error('Unrecognized identity key');
    return key;
  }
  return cachedIdentityKey(kid);
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
