const CACHE_NAME = 'anne-chatbot-v1';
const urlsToCache = [
    '/',
    '/assets/anne.jpg',
    '/assets/icon-192x192.png',
    '/assets/icon-512x512.png',
    '/manifest.json'
];

self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(urlsToCache))
    );
});

self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => {
                if (response) {
                    return response;
                }
                return fetch(event.request);
            })
    );
}); 