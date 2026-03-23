# Deploying Ruvisa (free / open-source)

Two parts:

1. **Frontend** — static files in `frontend/dist` → **Cloudflare Pages** (or Netlify).
2. **Backend** — FastAPI on port **8001** → run locally/VPS and expose with **Cloudflare Tunnel** or **ngrok**.

The app reads **`VITE_API_URL` at build time** (see `frontend/.env.example`). Dev uses Vite proxy `/api` → localhost; production needs a full HTTPS URL ending in `/api`.

## Run the API

```bash
pip install -r requirements-api.txt   # web + LangGraph + Ollama chat
# Full /analyze pipeline: use your full venv or requirements.txt (CV/YOLO).

python scripts/migrate_products_to_sqlite.py   # if needed (needs labeling data)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8001
```

## Expose the API (HTTPS)

**Cloudflare Quick Tunnel** (install [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/)):

```bash
cloudflared tunnel --url http://localhost:8001
```

Use the printed host as: `https://THAT-HOST/api` → set as `VITE_API_URL`.

**ngrok:** `ngrok http 8001` → `https://xxx.ngrok-free.app/api`.

## Build frontend

```bash
cd frontend
# Set VITE_API_URL in .env.production or export for one shot:
export VITE_API_URL=https://your-tunnel-host/api
npm ci && npm run build
```

`public/_redirects` enables SPA routing on Cloudflare Pages.

## Cloudflare Pages

- Connect Git with **root** `frontend`, build `npm ci && npm run build`, output **`dist`**.
- Add env var **`VITE_API_URL`** (production) = your tunnel URL including `/api`.
- Or upload: `npx wrangler pages deploy frontend/dist --project-name=ruvisa-frontend`

## Limits

- SQLite + uploads follow wherever the API runs; free tunnel URLs change unless you use a named tunnel/domain.
- See optional GitHub Action: `.github/workflows/deploy-frontend-pages.yml`.
