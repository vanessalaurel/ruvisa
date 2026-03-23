"""Pydantic models for database records."""

from datetime import datetime
from pydantic import BaseModel


class User(BaseModel):
    user_id: str
    name: str | None = None
    skin_type: str | None = None
    created_at: datetime | None = None


class Analysis(BaseModel):
    id: int | None = None
    user_id: str
    image_path: str | None = None
    concern_vector: list[float] | None = None
    acne_summary: dict | None = None
    wrinkle_summary: dict | None = None
    full_report: dict | None = None
    created_at: datetime | None = None


class Recommendation(BaseModel):
    id: int | None = None
    analysis_id: int | None = None
    user_id: str
    product_url: str | None = None
    product_title: str | None = None
    brand: str | None = None
    category: str | None = None
    similarity: float | None = None
    price: float | None = None
    created_at: datetime | None = None


class Purchase(BaseModel):
    id: int | None = None
    user_id: str
    product_url: str | None = None
    product_title: str | None = None
    price: float | None = None
    purchased_at: datetime | None = None
