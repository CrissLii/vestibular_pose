import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // 允许通过 Cloudflare Tunnel / ngrok 等外网域名访问开发服务器
    allowedHosts: true,
    host: true, // 允许局域网访问，如 http://192.168.1.13:5173
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/static': 'http://127.0.0.1:8000',
    },
  },
})
