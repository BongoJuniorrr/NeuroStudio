import {defineConfig} from 'vite';
export default defineConfig({build:{outDir:'../../src/neurostudio/static',emptyOutDir:true,rollupOptions:{output:{manualChunks:{charts:['recharts'],canvas:['@xyflow/react']}}}},server:{proxy:{'/api':'http://127.0.0.1:8765'}}});
