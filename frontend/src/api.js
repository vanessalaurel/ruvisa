/** Trailing/leading spaces in VITE_API_URL (e.g. pasted secrets) become %20 in paths and break routes. */
function apiBase() {
  const raw = import.meta.env.VITE_API_URL;
  if (raw === undefined || raw === null || String(raw).trim() === "") {
    return "/api";
  }
  return String(raw).trim().replace(/\/+$/, "");
}

const BASE = apiBase();

/**
 * Turn `/api/uploads/...` into a full URL when `VITE_API_URL` is set (e.g. Cloudflare Pages).
 * Relative URLs load against the site origin (*.pages.dev) and break for uploaded images.
 */
export function resolveApiMediaUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  const p = path.startsWith("/") ? path : `/${path}`;
  const raw = import.meta.env.VITE_API_URL;
  if (raw === undefined || raw === null || String(raw).trim() === "") {
    return p;
  }
  const base = apiBase();
  const suffix = p.replace(/^\/api/, "");
  return suffix.startsWith("/") ? `${base}${suffix}` : `${base}/${suffix}`;
}

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

export async function sendChat(userId, message) {
  return request("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, message }),
  });
}

export async function analyzeImage(userId, skinType, imageFile) {
  const form = new FormData();
  form.append("user_id", userId);
  form.append("skin_type", skinType);
  form.append("image", imageFile);
  return request("/analyze", { method: "POST", body: form });
}

export async function getRecommendations(userId, skinType, concernVector, budget, topN = 5) {
  let cv = concernVector;
  if (Array.isArray(cv)) {
    cv = cv.map((x) => {
      const n = Number(x);
      return Number.isFinite(n) ? n : 0;
    });
  }
  return request("/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId,
      skin_type: skinType,
      concern_vector: cv ?? null,
      budget: budget || null,
      top_n: topN,
    }),
  });
}

export async function listProducts(category, limit = 20, offset = 0) {
  const params = new URLSearchParams({ limit, offset });
  if (category) params.set("category", category);
  return request(`/products?${params}`);
}

export async function searchProducts(concern, skinType, maxPrice, minRating, sortBy = "evidence", limit = 10) {
  const params = new URLSearchParams({ concern, sort_by: sortBy, limit });
  if (skinType) params.set("skin_type", skinType);
  if (maxPrice) params.set("max_price", maxPrice);
  if (minRating) params.set("min_rating", minRating);
  return request(`/products/search?${params}`);
}

export async function getProductReviews(productUrl) {
  return request(`/products/reviews?product_url=${encodeURIComponent(productUrl)}`);
}

export async function getProfile(userId) {
  return request(`/profile/${userId}`);
}

export async function getHistory(userId) {
  return request(`/history/${userId}`);
}

export async function getJourney(userId) {
  return request(`/journey/${userId}`);
}

export async function recordPurchase(userId, product) {
  return request("/purchase", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId,
      product_url: product.product_url,
      product_title: product.title || product.product_title,
      price: product.price_value ?? (typeof product.price === "string" ? parseFloat(product.price?.replace(/[^0-9.]/g, "")) : product.price),
    }),
  });
}

export async function registerUser(name, email, password, skinType, concerns) {
  return request("/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, email, password, skin_type: skinType, concerns }),
  });
}

export async function loginUser(email, password) {
  return request("/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
}

export async function addToBag(userId, product) {
  return request("/bag/add", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId,
      product_url: product.product_url,
      product_title: product.title || product.product_title,
      brand: product.brand,
      price: product.price_value ?? (typeof product.price === "string" ? parseFloat(product.price?.replace(/[^0-9.]/g, "")) : product.price),
      image_url: product.image_url,
    }),
  });
}

export async function removeFromBag(userId, productUrl) {
  return request("/bag/remove", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, product_url: productUrl }),
  });
}

export async function getBag(userId) {
  return request(`/bag/${userId}`);
}

export async function toggleLike(userId, product) {
  return request("/like", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId,
      product_url: product.product_url,
      product_title: product.title || product.product_title,
      brand: product.brand,
      price: product.price_value ?? (typeof product.price === "string" ? parseFloat(product.price?.replace(/[^0-9.]/g, "")) : product.price),
      image_url: product.image_url,
    }),
  });
}

export async function getLiked(userId) {
  return request(`/liked/${userId}`);
}

export async function getTrending(limit = 10) {
  return request(`/trending?limit=${limit}`);
}

export async function updateSettings(userId, { name, email, currentPassword, newPassword }) {
  return request("/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId,
      name: name || undefined,
      email: email || undefined,
      current_password: currentPassword || undefined,
      new_password: newPassword || undefined,
    }),
  });
}

export async function healthCheck() {
  const res = await fetch(`${BASE.replace("/api", "")}/health`);
  return res.json();
}
