"""Pydantic request/response schemas for the API."""

from pydantic import BaseModel


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    skin_type: str | None = None
    concerns: list[str] | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    user_id: str
    name: str | None
    email: str | None
    skin_type: str | None
    concerns: list[str] | None = None


class ChatRequest(BaseModel):
    user_id: str
    message: str
    image_path: str | None = None


class ChatResponse(BaseModel):
    user_id: str
    response: str


class AnalyzeRequest(BaseModel):
    user_id: str
    image_path: str
    skin_type: str = "oily"


class AnalyzeResponse(BaseModel):
    user_id: str
    analysis_id: int
    concern_vector: list[float]
    acne_summary: dict
    wrinkle_summary: dict
    summary: str


class ProductSearchQuery(BaseModel):
    concern: str
    skin_type: str | None = None
    max_price: float | None = None
    min_rating: float | None = None
    sort_by: str = "evidence"
    limit: int = 10


class RecommendRequest(BaseModel):
    user_id: str
    skin_type: str
    concern_vector: list[float] | None = None
    budget: float | None = None
    top_n: int = 5


class ProfileResponse(BaseModel):
    user_id: str
    name: str | None
    skin_type: str | None
    created_at: str | None
    analyses_count: int
    purchases_count: int
    latest_analysis: dict | None
    recent_purchases: list[dict]


class HistoryResponse(BaseModel):
    user_id: str
    analyses: list[dict]
    purchases: list[dict]
    recommendations: list[dict]


class PurchaseRequest(BaseModel):
    user_id: str
    product_url: str
    product_title: str | None = None
    price: float | None = None


class BagRequest(BaseModel):
    user_id: str
    product_url: str
    product_title: str | None = None
    brand: str | None = None
    price: float | None = None
    image_url: str | None = None


class LikeRequest(BaseModel):
    user_id: str
    product_url: str
    product_title: str | None = None
    brand: str | None = None
    price: float | None = None
    image_url: str | None = None


class UpdateSettingsRequest(BaseModel):
    user_id: str
    name: str | None = None
    email: str | None = None
    current_password: str | None = None
    new_password: str | None = None
