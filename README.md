# Ruvisa

AI-assisted skincare app (FastAPI backend + Vite/React frontend).

- **Deploy / Cloudflare / ngrok:** [DEPLOY.md](./DEPLOY.md)
- **License:** [MIT](./LICENSE)
- **Backend (API + chat agent):** `pip install -r requirements-api.txt` — Ollama must be running for Ruvisa chat. Full face analysis (`/analyze`) needs the heavier [requirements.txt](./requirements.txt) stack (YOLO/CV) or your existing environment.
- **Frontend:** `cd frontend && npm ci && npm run dev`

### Push to GitHub (your machine)

This repo is initialized with `origin` → [github.com/vanessalaurel/ruvisa](https://github.com/vanessalaurel/ruvisa). From the project root:

```bash
git push -u origin main
```

Use **GitHub CLI** (`gh auth login`) or a **personal access token** when HTTPS asks for credentials; or switch the remote to SSH:

`git remote set-url origin git@github.com:vanessalaurel/ruvisa.git`
