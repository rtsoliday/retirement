// Google sign-in are handled by Firebase Authentication; Stripe sees only a verified UID.
const firebaseVersion = '12.0.0';
let auth;
let sdk;
let user = null;
let configured = false;
let providers = { google: false };
let chatgptSignedIn = false;

function providerFor(name) {
  if (name === 'google') return new sdk.GoogleAuthProvider();
  throw new Error('Unknown sign-in method');
}
export async function initializeSocialAuth(onChange) {
  const response = await fetch('/api/auth/config', { credentials: 'same-origin', cache: 'no-store' });
  if (!response.ok) throw new Error('Sign-in settings are unavailable');
  const config = await response.json();
  chatgptSignedIn = Boolean(config.chatgptSignedIn);
  providers = config.providers || providers;
  if (!config.configured) return;
  const [{ initializeApp }, authModule] = await Promise.all([
    import(`https://www.gstatic.com/firebasejs/${firebaseVersion}/firebase-app.js`),
    import(`https://www.gstatic.com/firebasejs/${firebaseVersion}/firebase-auth.js`),
  ]);
  sdk = authModule;
  auth = sdk.getAuth(initializeApp(config.firebase));
  await sdk.setPersistence(auth, sdk.browserLocalPersistence);
  configured = true;
  await new Promise((resolve, reject) => {
    let first = true;
    sdk.onAuthStateChanged(auth, next => {
      user = next;
      if (first) { first = false; resolve(); } else onChange?.();
    }, reject);
  });
}
export function socialState() {
  return { configured, chatgptSignedIn, signedIn: Boolean(user), accountKey: user ? `firebase:${user.uid}` : null, linkedProviders: user?.providerData.map(p => p.providerId) || [], enabledProviders: providers };
}
export async function authHeaders() {
  return user ? { Authorization: `Bearer ${await user.getIdToken()}` } : {};
}
export async function signInSocial(name) {
  if (!configured || !providers[name]) throw new Error('This sign-in method is not configured yet');
  await sdk.signInWithPopup(auth, providerFor(name));
}
export async function linkSocialProvider(name) {
  if (!user) throw new Error('Sign in with Google first');
  if (!providers[name]) throw new Error('This sign-in method is not configured yet');
  const provider = providerFor(name);
  if (user.providerData.some(p => p.providerId === provider.providerId)) return;
  await sdk.linkWithPopup(user, provider);
}
export async function signOutSocial() {
  if (auth) await sdk.signOut(auth);
}
