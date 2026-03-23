# Ruvisa

AI-assisted skincare app (FastAPI backend + Vite/React frontend).

- **Deploy / Cloudflare / ngrok:** [DEPLOY.md](./DEPLOY.md)
- **License:** [MIT](./LICENSE)
- **Backend (API + chat agent):** `pip install -r requirements-api.txt` — Ollama must be running for Ruvisa chat. Full face analysis (`/analyze`) needs the heavier [requirements.txt](./requirements.txt) stack (YOLO/CV) or your existing environment.
- **Frontend:** `cd frontend && npm ci && npm run dev`
