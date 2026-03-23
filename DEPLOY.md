# Deploying Ruvisa (Cloudflare Pages + API tunnel)

Repo: **[github.com/vanessalaurel/ruvisa](https://github.com/vanessalaurel/ruvisa)**

| Piece | Where it runs | Free option |
|--------|----------------|-------------|
| **Frontend** | Static files (`frontend/dist`) | **[Cloudflare Pages](https://pages.cloudflare.com/)** |
| **Backend** | FastAPI `:8001` | Your PC/VPS + **[cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/)** or [ngrok](https://ngrok.com/) |

`VITE_API_URL` is baked in at **build time** (see `frontend/.env.example`). It must be the full origin of the API **including** `/api`, e.g. `https://abc123.trycloudflare.com/api`.

---

## 1. Run the API locally

```bash
cd /path/to/ruvisa
pip install -r requirements-api.txt
# Optional: full /analyze (YOLO/CV) needs requirements.txt + models on that machine.

python scripts/migrate_products_to_sqlite.py   # if you use SQLite catalog
python -m uvicorn api.main:app --host 0.0.0.0 --port 8001
```

Sanity check: [http://127.0.0.1:8001/health](http://127.0.0.1:8001/health) → JSON with `status: ok`.

**Ollama** (Ruvisa chat): running on the same host as the API, default `http://localhost:11434`.

---

## 2. Expose the API over HTTPS (Quick Tunnel)

Install **cloudflared**, then:

```bash
cloudflared tunnel --url http://localhost:8001
```

Copy the `https://….trycloudflare.com` URL (changes each restart unless you use a [named tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/do-more-with-tunnels/local-management/tunnel-run/)).

Your frontend env value is:

```text
VITE_API_URL=https://<that-host>/api
```

**ngrok alternative:** `ngrok http 8001` → `VITE_API_URL=https://<subdomain>.ngrok-free.app/api`

---

## 3. Cloudflare Pages (GitHub)

### A) Classic Pages UI (root directory = `frontend`)

1. [Cloudflare Dashboard](https://dash.cloudflare.com/) → **Workers & Pages** → **Create** → **Pages** → **Connect to Git**.
2. Select **vanessalaurel/ruvisa**.
3. **Configure build:**

   | Setting | Value |
   |--------|--------|
   | Framework preset | None (or Vite if offered) |
   | Root directory | `frontend` |
   | Build command | `npm ci && npm run build` |
   | Build output directory | `dist` |

4. **Environment variables** → **Production** (and Preview):

   | Name | Example value |
   |------|----------------|
   | `VITE_API_URL` | `https://your-tunnel.trycloudflare.com/api` |

5. Save and deploy.

---

### B) Form with **Path `/`**, **Build command**, **Deploy command** (repo root)

Do **not** use `npx wrangler deploy` — that publishes a **Worker**, not your static Vite app.

Use **`wrangler pages deploy`** and build inside `frontend/`:

| Field | Value |
|--------|--------|
| **Project name** | `ruvisa` |
| **Build command** | `cd frontend && npm ci && npm run build` |
| **Deploy command** | `npx wrangler pages deploy frontend/dist --project-name=ruvisa` |
| **Non-production branch deploy command** | Leave **empty**, or use the same **Deploy command** if the form requires something |
| **Path** | `/` (repository root) |
| **Variable name** | `VITE_API_URL` |
| **Variable value** | `https://YOUR-TUNNEL.trycloudflare.com/api` (your real API URL + `/api`) |

**API token:** letting Cloudflare create one automatically is fine.

A root **`wrangler.toml`** in this repo sets `pages_build_output_dir = "frontend/dist"` for Wrangler; the explicit `frontend/dist` in the deploy command is what matters for CI.

**SPA routing:** `frontend/public/_redirects` → all routes to `index.html`.

---

## 4. CLI deploy (optional)

From `frontend/` after a production build:

```bash
cd frontend
export VITE_API_URL=https://your-tunnel-host/api
npm ci && npm run build
npx wrangler pages deploy dist
```

`wrangler.toml` in `frontend/` sets `name = "ruvisa"` and `pages_build_output_dir = "dist"`. Log in with `npx wrangler login` once.

---

## 5. After deploy — quick checks

- Open the Pages site → app loads without blank console errors.
- In DevTools **Network**, API calls go to your tunnel host (not `localhost`).
- `GET https://<api-host>/health` returns JSON (strip `/api` for health: same host, path `/health`).

---

## 6. Production hardening (later)

- In `api/main.py`, replace `allow_origins=["*"]` with your exact Pages URL(s).
- Use a **stable** API hostname (named Cloudflare Tunnel or VPS + domain).
- Do not commit `.env`, tokens, or `data/skincare.db` with real users (see `.gitignore`).

---

## 7. GitHub Action

Optional automated deploy: [.github/workflows/deploy-frontend-pages.yml](.github/workflows/deploy-frontend-pages.yml)  
Secrets: `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `VITE_API_URL`. Trigger **Actions → Run workflow**.

---

## Limits (honest)

- **Quick Tunnel** URLs change when the tunnel restarts → rebuild Pages or update `VITE_API_URL` and redeploy.
- **SQLite + uploads** live on whatever machine runs `uvicorn`; Pages only hosts the UI.
