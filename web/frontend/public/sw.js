// Unregister old service worker and clear caches
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", async () => {
  const keys = await caches.keys();
  await Promise.all(keys.map((k) => caches.delete(k)));
  const clients = await self.clients.matchAll({ type: "window" });
  for (const client of clients) client.navigate(client.url);
  await self.registration.unregister();
});
