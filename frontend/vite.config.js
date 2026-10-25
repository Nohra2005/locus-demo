import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: "autoUpdate",
      manifest: false,
      workbox: {
        globPatterns: ["**/*.{js,css,html,ico,png,svg,woff2}"],
        runtimeCaching: [
          {
            urlPattern: /\/static\/.*/i,
            handler: "CacheFirst",
            options: {
              cacheName: "locus-product-images",
              expiration: { maxEntries: 200, maxAgeSeconds: 604800 },
            },
          },
          {
            urlPattern: /^https:\/\/.*\.basemaps\.cartocdn\.com\/.*/i,
            handler: "CacheFirst",
            options: {
              cacheName: "locus-map-tiles",
              expiration: { maxEntries: 500, maxAgeSeconds: 2592000 },
            },
          },
        ],
      },
    }),
  ],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/detect":          { target: "http://localhost:8000", changeOrigin: true },
      "/search":          { target: "http://localhost:8000", changeOrigin: true },
      "/add":             { target: "http://localhost:8000", changeOrigin: true },
      "/feedback":        { target: "http://localhost:8000", changeOrigin: true },
      "/static":          { target: "http://localhost:8000", changeOrigin: true },
      "/judge-scores":    { target: "http://localhost:8000", changeOrigin: true },
      "/classify-crop":   { target: "http://localhost:8000", changeOrigin: true },
      "/refine":          { target: "http://localhost:8000", changeOrigin: true },
      "/search/":         { target: "http://localhost:8000", changeOrigin: true },
      "/auth":            { target: "http://localhost:8000", changeOrigin: true },
      "/store-catalogue": { target: "http://localhost:8000", changeOrigin: true },
      "/store-stats":     { target: "http://localhost:8000", changeOrigin: true },
      "/add-bulk-batch":  { target: "http://localhost:8000", changeOrigin: true },
      "/scrape":              { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});