// Service worker: keep the app shell available offline without ever serving a stale UI.
// Network-first for the shell and static files (cache fallback when the server is
// unreachable); API and page responses are never cached, so results, the dashboard
// and the sample strip are always live. Bump SHELL to drop old caches.
const SHELL = 'mcv-shell-v16';
const ASSETS = ['/', '/static/theme.css', '/static/manifest.webmanifest', '/static/icon.svg'];
const cacheable = p => p === '/' || p.startsWith('/static/');
self.addEventListener('install', e => { e.waitUntil(caches.open(SHELL).then(c => c.addAll(ASSETS)).catch(() => {}).then(() => self.skipWaiting())); });
self.addEventListener('activate', e => { e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== SHELL).map(k => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.origin !== self.location.origin || !cacheable(url.pathname)) return;
  e.respondWith(
    fetch(e.request).then(res => {
      if (res.ok) { const copy = res.clone(); caches.open(SHELL).then(c => c.put(e.request, copy)); }
      return res;
    }).catch(() => caches.match(e.request, { ignoreSearch: url.pathname === '/' }).then(hit => hit || Response.error()))
  );
});
