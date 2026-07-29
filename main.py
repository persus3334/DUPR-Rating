"""
DUPR proxy backend — per-user authentication edition.

Design goals (learned the hard way):
  1. Every user logs in with THEIR OWN DUPR account. Their password passes
     through this server for exactly one request and is never stored or
     logged. Only the bearer token goes back to the browser.
  2. This server stores NOTHING sensitive. No database, no credentials,
     no tokens. If the box is compromised, there's nothing to steal.
  3. Aggressive response caching (rating history only changes when matches
     post), so 1,000 users in 3 hours ≈ a few hundred upstream requests,
     not hundreds of thousands.
  4. Per-IP rate limiting so nobody can hammer DUPR through you.

Run:
    pip install fastapi httpx uvicorn cachetools
    uvicorn main:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000/
"""

import asyncio
import time
from datetime import datetime
from collections import defaultdict, deque

import httpx
from cachetools import TTLCache
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

DUPR_BASE = "https://api.dupr.gg"
API_VER = "v1.0"

# Headers DUPR's backend expects (browsers can't set Origin/Referer
# cross-site, which is why this proxy has to exist at all).
UPSTREAM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Origin": "https://dashboard.dupr.com",
    "Referer": "https://dashboard.dupr.com/",
    "Content-Type": "application/json",
}

# ---------------------------------------------------------------------------
# Caching + rate limiting (in-memory; swap for Redis if you scale out)
# ---------------------------------------------------------------------------
# Rating/match/search data is identical no matter whose token fetched it,
# so it's safe to cache by player, not by user.
cache = TTLCache(maxsize=10_000, ttl=3600)          # 1 hour
search_cache = TTLCache(maxsize=10_000, ttl=86400)  # DUPR-ID→numeric-ID rarely changes

RATE_LIMIT = 30          # requests
RATE_WINDOW = 60         # per seconds, per IP
_hits: dict[str, deque] = defaultdict(deque)
_lock = asyncio.Lock()


async def check_rate_limit(request: Request):
    # Behind a hosting proxy (Render/Fly/nginx), the real client IP arrives
    # in X-Forwarded-For; request.client.host would be the proxy itself.
    ip = (request.headers.get("x-forwarded-for", "").split(",")[0].strip()
          or (request.client.host if request.client else "unknown"))
    now = time.monotonic()
    async with _lock:
        q = _hits[ip]
        while q and now - q[0] > RATE_WINDOW:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            raise HTTPException(429, "Slow down — try again in a minute.")
        q.append(now)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
from contextlib import asynccontextmanager

client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app):
    global client
    client = httpx.AsyncClient(base_url=DUPR_BASE, headers=UPSTREAM_HEADERS, timeout=20)
    yield
    await client.aclose()


app = FastAPI(title="DUPR Stats Proxy", docs_url=None, redoc_url=None,
              lifespan=lifespan)


def bearer(authorization: str | None) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing bearer token — log in first.")
    return {"Authorization": authorization}


async def upstream(method: str, path: str, *, headers=None, json=None):
    """Forward a request to DUPR and translate failures cleanly."""
    try:
        r = await client.request(method, path, headers=headers, json=json)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"DUPR unreachable: {e.__class__.__name__}")
    if r.status_code == 401 or r.status_code == 403:
        raise HTTPException(401, "DUPR rejected the token — log in again.")
    if r.status_code != 200:
        # Surface what DUPR actually said (status + body snippet) so failures
        # are debuggable. Response bodies here are error JSON, never secrets.
        snippet = r.text[:300].replace("\n", " ")
        print(f"[upstream] {method} {path} -> {r.status_code}: {snippet}")
        raise HTTPException(r.status_code, f"DUPR error {r.status_code}: {snippet}")
    return r.json()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
class LoginBody(BaseModel):
    email: str
    password: str


@app.post("/api/login")
async def login(body: LoginBody, request: Request):
    """
    Credential passthrough. The password lives in memory for the duration
    of this one request. It is not logged, not stored, not cached.
    """
    await check_rate_limit(request)
    data = await upstream(
        "POST", f"/auth/{API_VER}/login/",
        json={"email": body.email, "password": body.password},
    )
    result = data.get("result", {})
    token = result.get("accessToken")
    if not token:
        raise HTTPException(401, "Login failed — check email/password.")
    # Return ONLY what the client needs. Token lives in the browser.
    return {"accessToken": token, "fullName": result.get("user", {}).get("fullName")}


@app.get("/api/profile")
async def profile(request: Request, authorization: str | None = Header(None)):
    await check_rate_limit(request)
    data = await upstream("GET", f"/user/{API_VER}/profile/",
                          headers=bearer(authorization))
    return data.get("result", {})


@app.get("/api/player/{dupr_id}")
async def find_player(dupr_id: str, request: Request,
                      authorization: str | None = Header(None)):
    """Resolve a human DUPR ID (e.g. NRRGJZ) to numeric ID + name."""
    await check_rate_limit(request)
    key = dupr_id.upper()
    if key in search_cache:
        return search_cache[key]

    # Page through results 10 at a time (mirrors the DUPR website's own
    # behavior; their API rejects limits over 25). Cap at 100 results.
    offset = 0
    while offset is not None and offset < 100:
        data = await upstream(
            "POST", f"/player/{API_VER}/search",
            headers=bearer(authorization),
            json={
                "limit": 10, "offset": offset, "query": key, "exclude": [],
                "includeUnclaimedPlayers": True,
                "filter": {"lat": 33.75, "lng": -84.39,
                           "rating": {"maxRating": None, "minRating": None},
                           "locationText": ""},
            },
        )
        result = data.get("result", {})
        for hit in result.get("hits", []):
            if (hit.get("duprId") or "").upper() == key:
                out = {"id": hit.get("id"), "fullName": hit.get("fullName"),
                       "duprId": key, "ratings": hit.get("ratings")}
                search_cache[key] = out
                return out
        total = result.get("total", 0)
        limit = result.get("limit", 10)
        offset = offset + limit if offset + limit < total else None

    # DUPR's search excludes the logged-in account from its own results,
    # so a user searching their own ID never gets a hit. Fall back to
    # their profile. The profile's DUPR-ID field name is unreliable
    # (sometimes absent), so match the value against every string field.
    me = await upstream("GET", f"/user/{API_VER}/profile/",
                        headers=bearer(authorization))
    prof = me.get("result", {})
    if any(isinstance(v, str) and v.strip().upper() == key
           for v in prof.values()):
        out = {"id": prof.get("id"), "fullName": prof.get("fullName"),
               "duprId": key, "ratings": prof.get("ratings")}
        search_cache[key] = out
        return out
    print(f"[debug] self-lookup miss for {key}; profile fields: {sorted(prof.keys())}")

    raise HTTPException(404, f"No player found with DUPR ID {key}")


@app.get("/api/rating-history/{player_id}")
async def rating_history(player_id: int, request: Request, type: str = "DOUBLES",
                         authorization: str | None = Header(None)):
    await check_rate_limit(request)
    fmt = type.upper()
    if fmt not in ("DOUBLES", "SINGLES"):
        raise HTTPException(400, "type must be DOUBLES or SINGLES")

    key = ("rh", player_id, fmt)
    if key in cache:
        return cache[key]

    # Date-windowed pagination at limit 100 (proven approach: this endpoint
    # accepts 100 per page; only /search caps at 25). Stop when a page comes
    # back shorter than the limit. 2020 predates DUPR itself.
    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    history, full_name, offset, limit = [], None, 0, 100
    while True:
        data = await upstream(
            "POST", f"/player/{API_VER}/{player_id}/rating-history",
            headers=bearer(authorization),
            json={"startDate": "2020-01-01", "endDate": end_date,
                  "limit": limit, "offset": offset,
                  "sortBy": "asc", "type": fmt},
        )
        result = data.get("result", {})
        if not full_name:
            full_name = result.get("fullName")
        page = result.get("ratingHistory", []) or []
        history.extend(page)
        if len(page) < limit or len(history) > 20000:
            break
        offset += limit
        await asyncio.sleep(0.2)  # be polite

    out = {"playerId": player_id, "type": fmt,
           "fullName": full_name, "history": history}
    cache[key] = out
    return out


@app.get("/api/match-history/{player_id}")
async def match_history(player_id: int, request: Request,
                        authorization: str | None = Header(None)):
    await check_rate_limit(request)
    key = ("mh", player_id)
    if key in cache:
        return cache[key]

    hits, offset, limit = [], 0, 25
    while len(hits) < 5000:
        data = await upstream(
            "POST", f"/player/{API_VER}/{player_id}/history",
            headers=bearer(authorization),
            json={"filters": {"eventFormat": None},
                  "sort": {"order": "DESC", "parameter": "MATCH_DATE"},
                  "limit": limit, "offset": offset},
        )
        result = data.get("result", {})
        page = result.get("hits", []) or []
        hits.extend(page)
        # Prefer the hasMore flag; fall back to total/offset math.
        if "hasMore" in result:
            more = bool(result.get("hasMore")) and page
        else:
            total = result.get("total", 0)
            more = offset + limit < total
        if not more:
            break
        offset += limit
        await asyncio.sleep(0.2)  # be polite

    out = {"playerId": player_id, "matches": hits}
    cache[key] = out
    return out


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.exception_handler(HTTPException)
async def http_exc(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
