"""FastAPI application for the skincare advisory system."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import db.crud as crud
from db.database import init_db
from .routes import router

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    np, nr = crud.count_catalog_rows()
    logger.info("Catalog (SQLite): %d products, %d review-score rows", np, nr)
    if np == 0:
        logger.warning(
            "Product catalog is empty. Populate with: python scripts/migrate_products_to_sqlite.py"
        )
    yield


app = FastAPI(
    title="Skincare Advisory Agent",
    description="AI-powered skincare analysis and product recommendation system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
app.mount("/api/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


@app.get("/health")
async def health():
    np, nr = crud.count_catalog_rows()
    return {
        "status": "ok",
        "service": "skincare-agent",
        "catalog_products": np,
        "catalog_review_scores": nr,
    }
