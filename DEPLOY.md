# Deploying Ruvisa (Cloudflare Pages + API tunnel)

Repo: **[github.com/vanessalaurel/ruvisa](https://github.com/vanessalaurel/ruvisa)**

| Piece | Where it runs | Free option |
|--------|----------------|-------------|
| **Frontend** | Static files (`frontend/dist`) | **[Vercel](https://vercel.com/)**, **[Netlify](https://netlify.com/)**, or [Cloudflare Pages](https://pages.cloudflare.com/) |
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

If logs repeat **`Failed to dial a quic connection`** / **`CRYPTO_ERROR 0x178`** / **`tls: no application protocol`**, your network is blocking or breaking QUIC. Use TCP instead:

```bash
cloudflared tunnel --url http://localhost:8001 --protocol http2
```

(The **`ICMP proxy`** / **`ping_group_range`** messages are usually harmless for a quick tunnel.)

Copy the `https://….trycloudflare.com` URL (changes each restart unless you use a [named tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/do-more-with-tunnels/local-management/tunnel-run/)).

Your frontend env value is:

```text
VITE_API_URL=https://<that-host>/api
```

**ngrok alternative:** `ngrok http 8001` → `VITE_API_URL=https://<subdomain>.ngrok-free.app/api`

---

## 2b. Easier than Cloudflare Pages: Vercel or Netlify (frontend only)

**ngrok does not host your website** — it only gives a public URL to **localhost** (same job as cloudflared for the API). To avoid Wrangler / Workers Builds / API tokens, put the **Vite app** on a host that builds from GitHub in a few clicks:

### Vercel

1. Sign up at **[vercel.com](https://vercel.com)** → **Add New** → **Project** → import **ruvisa**.
2. **Root Directory:** `frontend` → **Framework Preset:** Vite (or Other).
3. **Build Command:** `npm run build` · **Output Directory:** `dist`
4. **Environment Variables:** `VITE_API_URL` = `https://YOUR-NGROK-OR-TUNNEL-HOST/api`
5. Deploy. Redeploy when your tunnel URL changes (or use a stable VPS API later).

### Netlify

1. **[netlify.com](https://www.netlify.com)** → **Add new site** → **Import from Git** → pick the repo.
2. **Base directory:** `frontend` · **Build command:** `npm run build` · **Publish directory:** `dist` (output of Vite inside `frontend/`).
3. Add **`VITE_API_URL`** in **Site configuration → Environment variables**.

### API tunnel (pick one)

| Tool | Command | Free tier notes |
|------|---------|-----------------|
| **[ngrok](https://ngrok.com/)** | Install CLI, `ngrok config add-authtoken …`, then `ngrok http 8001` | Needs account; URL may change on free plan |
| **cloudflared** | `cloudflared tunnel --url http://localhost:8001 --protocol http2` | No account for quick tunnels |

Keep **`uvicorn`** running on **8001** while you demo; set **`VITE_API_URL`** on Vercel/Netlify to `https://<public-host>/api`.

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

   **Do not** add `pip install -r requirements.txt` (or any Python step) to the Pages build. That file pulls PyTorch/CUDA, YOLO, spaCy, etc. — multi‑GB and will fail with **`No space left on device`** on Pages. The static site only needs **Node** in `frontend/`. Python deps run on **your machine or a VPS** for the API (`requirements-api.txt` or full `requirements.txt` there only).

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
| **Deploy command** | Prefer **empty** — if the UI offers **build output directory** / automatic upload, set output to `frontend/dist` and skip Wrangler here. Otherwise: `npx wrangler pages deploy frontend/dist --project-name=ruvisa` |
| **Non-production branch deploy command** | Leave **empty**, or use the same **Deploy command** if the form requires something |
| **Path** | `/` (repository root) |
| **Variable name** | `VITE_API_URL` |
| **Variable value** | `https://YOUR-TUNNEL.trycloudflare.com/api` (your real API URL + `/api`) |

**If deploy uses Wrangler** (`npx wrangler pages deploy …`), error **`Authentication error [code: 10000]`** on `/pages/projects/...` means the token can reach “who am I” but **cannot manage Pages**. Workers Builds’ **auto-generated** token includes Workers/KV/R2 — it often does **not** include **Cloudflare Pages**.

Create a **new** [API token](https://dash.cloudflare.com/profile/api-tokens) → **Create Custom Token** with **all** of these (same account as in **Workers & Pages** → your account id in the sidebar / URL):

| Permission group | Access |
|------------------|--------|
| **Account** → **Cloudflare Pages** | **Edit** |
| **Account** → **Workers Scripts** | **Edit** |
| **Account** → **Account Settings** | **Read** |
| **User** → **User Details** | **Read** |
| **User** → **Memberships** | **Read** |

**Account resources:** Include → **your** account (not “All accounts” unless you intend that). No **Client IP Address Filtering** unless you know you need it.

Paste that token into **`CLOUDFLARE_API_TOKEN`** (build/deploy variables for this project), save, redeploy.

**Optional check on your PC** (replace `TOKEN`):

```bash
curl -sS "https://api.cloudflare.com/client/v4/accounts/YOUR_ACCOUNT_ID/pages/projects" \
  -H "Authorization: Bearer TOKEN" | head -c 400
```

If you see **`"success":true`**, Pages scope is OK. If you see **10000** or **Forbidden**, the token still lacks **Cloudflare Pages → Edit**.

A root **`wrangler.toml`** in this repo sets `pages_build_output_dir = "frontend/dist"` for Wrangler; the explicit `frontend/dist` in the deploy command is what matters for CI.

**SPA routing:** `frontend/public/_redirects` → all routes to `index.html`.

### Worker vs Pages (same name, different product)

A **Worker** connected to Git (what you see under `workers/services/view/…`) is **not** a **Pages** project. `wrangler pages deploy --project-name=…` only talks to **Pages**. Error **`Project not found [code: 8000007]`** means there is no **Pages** project with that name yet.

### No “Create Pages” button in the dashboard?

Create the Pages project from your machine (Wrangler):

```bash
export CLOUDFLARE_API_TOKEN="your-token-with-pages-edit"
export CLOUDFLARE_ACCOUNT_ID="your-account-id"
npx wrangler pages project create ruvisa-site
```

Then deploy with `--project-name=ruvisa-site`. If a **Worker** is already named `ruvisa`, use a **different** Pages name (e.g. `ruvisa-site`) so names stay clear.

With **Root directory** = `frontend`, deploy command should be `npx wrangler pages deploy dist --project-name=ruvisa-site` (not `frontend/dist`).

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

## 7. When Cloudflare’s Git build keeps failing (auth `10000`)

Workers Builds can only use tokens Cloudflare creates (“**only API can edit**”). Those often **do not** include **Cloudflare Pages → Edit**, so `wrangler pages deploy` fails even though you’re “Super Administrator” in the dashboard (dashboard role ≠ API token scopes).

**Still on Cloudflare Pages** — just deploy from **GitHub Actions** instead of Cloudflare’s build worker:

1. **[Create a custom API token](https://dash.cloudflare.com/profile/api-tokens)** with **Account → Cloudflare Pages → Edit**, **Workers Scripts → Edit**, **Account Settings → Read**, **User → User Details → Read**, **User → Memberships → Read** (same table as §3B). Account = yours.

2. **GitHub** → repo **Settings → Secrets and variables → Actions → New repository secret**:

   | Secret | Value |
   |--------|--------|
   | `CLOUDFLARE_API_TOKEN` | paste the custom token |
   | `CLOUDFLARE_ACCOUNT_ID` | from Cloudflare sidebar / URL (e.g. `6d411d62…`) |
   | `VITE_API_URL` | `https://your-tunnel.trycloudflare.com/api` |

3. **Pages project name** must match the workflow: **`ruvisa`** (see [deploy-frontend-pages.yml](.github/workflows/deploy-frontend-pages.yml)). In **Workers & Pages → Pages**, create or rename the project to **`ruvisa`**, or change `--project-name=` in the workflow to match your real project name.

4. **Actions** → **Deploy frontend to Cloudflare Pages** → **Run workflow**, or push to **`main`** (workflow also runs on pushes that touch `frontend/`).

5. **Optional:** In Cloudflare, turn off or ignore the **connected Git** build for this app so you’re not debugging two pipelines at once.

---

## Limits (honest)

- **Quick Tunnel** URLs change when the tunnel restarts → rebuild Pages or update `VITE_API_URL` and redeploy.
- **SQLite + uploads** live on whatever machine runs `uvicorn`; Pages only hosts the UI.
