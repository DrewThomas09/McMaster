// Minimal service worker: cache the app shell so the page opens instantly (and
// offline shows the shell); identification itself always needs the server.
const SHELL = 'mcv-shell-v2';
const ASSETS = ['/', '/static/theme.css', '/static/manifest.webmanifest', '/static/icon.svg'];
self.addEventListener('install', e => { e.waitUntil(caches.open(SHELL).then(c => c.addAll(ASSETS)).then(() => self.skipWaiting())); });
self.addEventListener('activate', e => { e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== SHELL).map(k => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.pathname.startsWith('/identify') || url.pathname.startsWith('/parts/') || url.pathname.startsWith('/search') || url.pathname.startsWith('/status') || url.pathname.startsWith('/metrics')) return;
  e.respondWith(caches.match(e.request).then(hit => hit || fetch(e.request).then(res => { const copy = res.clone(); caches.open(SHELL).then(c => c.put(e.request, copy)); return res; }).catch(() => hit)));
});
