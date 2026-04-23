"""
Fashion Advisor API
====================
RAG-powered fashion advice grounded in Die, Workwear! (Derek Guy's menswear blog).

Endpoints:
    POST /advice       - Text-based fashion advice
    POST /outfit-check - Upload an outfit photo for critique (image-only)
    POST /shop         - Shopping recommendations at a given budget
    POST /shop-vibe    - Structured shopping recs based on a vibe/occasion
    GET  /images/{filename} - Serve scraped blog images

Run:
    uvicorn app:app --reload --port 8000
"""

import os
import io
import base64
import json
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import anthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fashion-advisor")
from pinecone import Pinecone
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from typing import Optional, List
from pydantic import BaseModel
from PIL import Image
from serpapi import GoogleSearch
import requests as _requests
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

_executor = ThreadPoolExecutor(max_workers=8)

CLAUDE_MODEL = "claude-sonnet-4-20250514"
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "eaf631bccd0471c91ad21097a4c78eb76978dabbc4b88a76825b4d2564c00b6f")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pcsk_2PdVrk_CiVFvEbYqs8ts2GqJ69PMurqFpkeJDBjA1JdoPCzGJ6xvovgtn8jxLHyy5STAbH")
PINECONE_INDEX = "style-advice"
TOP_K = 12  # number of chunks to retrieve

# Initialize Pinecone at module level (no heavy ML model needed!)
_pc = Pinecone(api_key=PINECONE_API_KEY)
_pinecone_index = _pc.Index(PINECONE_INDEX)
EMBED_MODEL = "multilingual-e5-large"
IMAGES_DIR = os.environ.get("IMAGES_DIR", os.path.expanduser("~/Downloads/images"))
POSTS_JSON = os.environ.get("POSTS_JSON", os.path.expanduser("~/Downloads/dieworkwear_posts.json"))

# Curated catalog extracted from TikTok menswear creators. Loaded once at
# startup. Treated as brand+category EVIDENCE (not a contract for specific
# products) so it stays useful as items get discontinued or renamed — Claude
# uses it to know which brands are vouched for in which lanes/categories,
# and search_product hunts for the brand's current matching item.
_TIKTOK_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "data", "tiktok_catalog.json")
TIKTOK_CATALOG = []
try:
    with open(_TIKTOK_CATALOG_PATH) as _f:
        TIKTOK_CATALOG = json.load(_f).get("records", [])
    logger.info("Loaded TikTok catalog: %d records from %s",
                len(TIKTOK_CATALOG), _TIKTOK_CATALOG_PATH)
except FileNotFoundError:
    logger.warning("TikTok catalog not found at %s — running without curated brand evidence",
                   _TIKTOK_CATALOG_PATH)


# Brands we deliberately don't recommend, even when they're in the curated
# catalog. Currently used to suppress brands whose own website has template
# behavior that consistently produces wrong-image / wrong-variant cards
# (Todd Snyder's Shopify variant-default issue). Match is case-insensitive.
BLOCKED_CATALOG_BRANDS = {"todd snyder"}


def _is_blocked_brand(brand_str) -> bool:
    if not brand_str:
        return False
    return str(brand_str).strip().lower() in BLOCKED_CATALOG_BRANDS


def format_catalog_for_prompt() -> str:
    """Render the curated catalog as a brand+category evidence section for
    the outfit-composition prompt. Groups by lane → category, lists brands
    with example items and prices. The price ladders get a dedicated section
    so Claude can pick the right brand for the user's budget tier.

    Catalog records whose brand is in BLOCKED_CATALOG_BRANDS are stripped
    from both sections so Claude doesn't see them as options."""
    if not TIKTOK_CATALOG:
        return ""

    # Group product cards: lane -> category -> [items], skipping blocked brands
    by_lane: dict = {}
    for r in TIKTOK_CATALOG:
        if r.get("type") != "product_card":
            continue
        if _is_blocked_brand(r.get("brand")):
            continue
        lane = r.get("lane_guess") or "uncategorized"
        cat = r.get("category") or "other"
        by_lane.setdefault(lane, {}).setdefault(cat, []).append(r)

    lines = ["═══ CURATED BRAND CATALOG ═══",
             "Brands and items vetted by trusted menswear curators. Use these as a "
             "STRONG PREFERENCE when they fit the vibe. Treat catalog items as "
             "evidence that the brand makes great pieces in this category — you "
             "may commit to the brand even if a different specific product is more "
             "appropriate for the vibe (the system will search the brand's site for "
             "the current matching item).",
             ""]

    for lane in sorted(by_lane):
        lines.append(f"LANE: {lane}")
        for cat in sorted(by_lane[lane]):
            items = by_lane[lane][cat]
            lines.append(f"  {cat}:")
            for it in items:
                price = it.get("price")
                price_str = f"${price}" if price else "?"
                brand = it.get("brand") or "?"
                name = it.get("name") or "?"
                lines.append(f"    - {brand} — {name} ({price_str})")
        lines.append("")

    # Price-ladder section: best for budget-aware picking. Drop any blocked
    # brand entries from each ladder; if a ladder ends up with no entries,
    # skip the line entirely.
    ladders = [r for r in TIKTOK_CATALOG if r.get("type") == "price_ladder"]
    if ladders:
        lines.append("═══ PRICE-TIER BRAND MAP (for budget-aware picking) ═══")
        for ld in ladders:
            cat_name = ld.get("category_name") or ld.get("category") or "?"
            opts = [o for o in (ld.get("options") or [])
                    if not _is_blocked_brand(o.get("brand"))]
            if not opts:
                continue
            opts_str = " / ".join(
                f"{o.get('brand')} (${o.get('price')}, {o.get('tier')})"
                for o in opts
            )
            lines.append(f"{cat_name}: {opts_str}")
        lines.append("")

    return "\n".join(lines)


CATALOG_PROMPT_TEXT = format_catalog_for_prompt()
logger.info("Catalog prompt block: %d chars", len(CATALOG_PROMPT_TEXT))

# ── Build image index: map post titles to their local image filenames ────────
IMAGE_INDEX = {}  # title -> [{"file": "post_0001_img_00.jpg", "url": "https://..."}]
try:
    with open(POSTS_JSON, "r") as f:
        _posts_data = json.load(f)
    for post in _posts_data["posts"]:
        title = post.get("title", "")
        local_imgs = post.get("local_images", [])
        remote_imgs = post.get("image_urls", [])
        if title and local_imgs:
            IMAGE_INDEX[title] = [
                {"file": os.path.basename(li), "url": remote_imgs[i] if i < len(remote_imgs) else ""}
                for i, li in enumerate(local_imgs)
            ]
except Exception:
    pass

SYSTEM_PROMPT = """You are a menswear style advisor whose taste and knowledge is grounded in the writings of Derek Guy (Die, Workwear! blog). You give practical, opinionated fashion advice that balances classic style principles with modern sensibility.

Core principles you follow (derived from the corpus):
- Fit is paramount. Clothes should follow the body's natural lines without being too tight or too loose.
- Quality over quantity. A smaller wardrobe of well-made pieces beats a closet full of fast fashion.
- Context matters. Dress for the occasion, the setting, and the culture — not just "the rules."
- Personal style develops through understanding why things work, not just copying outfits.
- Price doesn't always equal quality. There are great options at every budget level.
- Details matter — fabric, construction, proportion — but shouldn't become fetishistic.
- Style should feel natural, not costumey. The best-dressed person in the room shouldn't look like they're trying hardest.

When answering:
- Reference specific insights from the provided blog excerpts when relevant.
- Be direct and opinionated — Derek doesn't hedge, and neither should you.
- Provide concrete recommendations (specific items, brands, approaches), not vague platitudes.
- If critiquing an outfit, be honest but constructive. Explain WHY something works or doesn't.
- For shopping recommendations, always consider value — the best $50 shirt matters as much as the best $500 one.
- Acknowledge when something is subjective or when multiple valid approaches exist.
"""

SHOP_SYSTEM_PROMPT = SYSTEM_PROMPT + """
IMPORTANT: When recommending specific brands or products, include direct links to specific product pages whenever possible. Format links as markdown: [Product Name](https://url). For example, link to the exact shoe on a retailer's site, not just the brand homepage. Use well-known retailers like Mr Porter, END Clothing, Sid Mashburn, No Man Walks Alone, J.Crew, Uniqlo, etc. and link to their product search or category pages for the specific item type. Only fall back to a brand's main website if you truly cannot construct a more specific link.
"""

VIBE_ITEMS_PROMPT = SYSTEM_PROMPT + """
You are helping a user shop for a complete outfit based on a vibe or occasion they describe.

You MUST respond with ONLY a valid JSON array of 6-8 objects. No markdown, no explanation, no preamble — just the JSON array.

Each object must have:
- "item": a specific, searchable garment or accessory description (style, color, fabric, fit). Be specific enough that searching for it on a retailer's site would find the right kind of product.
- "slot": one of "outerwear", "top", "bottom", "shoes", or "accessory" — indicates what part of the outfit this is.
- "brand": when possible, commit to a specific brand — STRONGLY PREFER brands from the curated catalog provided in the user message (or brands clearly adjacent to that lane). Use the catalog as evidence that a brand is good in this category, even if the specific product you'd commission from them is different from the catalog example. If no catalog brand fits and you don't have a confident pick, return an empty string "" and the system will broaden the search.

Cover a full outfit: outerwear (jacket/blazer/coat — skip if not needed for the vibe), top (shirt/sweater/polo), bottom (pants/shorts), shoes, and 1-2 accessories (belt, watch, sunglasses, etc).

═══════════════════════════════════════════════════
TASTE — read this twice
═══════════════════════════════════════════════════
The output should be a normal, tasteful outfit a real person with good taste would actually wear. NOT a costume, NOT an editorial styling, NOT a "themed" interpretation, NOT a Pitti Uomo street-style shoot. Think "well-dressed friend getting dressed for this thing."

NEVER recommend:
- Boat shoes / deck shoes / Sperry-style topsiders. Ever. For any vibe.
- Swim trunks / board shorts / swim shorts unless the vibe is EXPLICITLY beach, pool, boat, or surf. Outdoor concert is not a beach. Picnic is not a beach.
- Loafers for outdoor sports, concerts, hiking, athletic activities, or anything where you'd be on your feet for hours.
- Novelty / costume pieces (huaraches, bolo ties, fedoras, capri-length anything, tropical M-65 coats, Portuguese flannel camp shirts, ascots, monocles).
- Avant-garde / runway-y labels (Junya, CDG, Yohji, Rick Owens, Bode for clothing, Visvim, Story mfg, Engineered Garments) unless the user explicitly asks for designer/avant-garde.
- Sunglasses with cat-eye, oversized round, or other shapes that read as women's frames. Stick to classic shapes (aviator, wayfarer, clubmaster, square) for menswear.

═══════════════════════════════════════════════════
SCENARIO SPECIFICS
═══════════════════════════════════════════════════
- Outdoor concert / live music / festival / brewery / coffee shop weekend → casual tee or short-sleeve shirt or polo, chinos / jeans / casual shorts (NOT swim), low-top sneakers (NOT loafers, NOT boat shoes), sunglasses optional.
- Beach / pool / boat (explicit aquatic vibes only) → swim trunks, rashguard or tee, slides or sandals.
- Gym / workout / running → performance top, athletic shorts or joggers, trainers.
- Office / business casual → chinos or wool trousers, oxford or polo, leather sneakers or loafers.
- Wedding (non-themed) → suit or odd jacket + trousers, dress shirt, dress shoes.
- Cool evening out → light jacket or sweater fine, jeans or chinos, sneakers or loafers.
- Warm-weather casual → no jackets, no hats unless user asked for one.
- First day of [job/school] → smart casual, slightly overdressed beats underdressed.

Example output for "outdoor concert in Miami":
[{"item": "white pocket t-shirt heavyweight cotton", "slot": "top", "brand": "Buck Mason"}, {"item": "navy lightweight cotton chino shorts 7 inch inseam", "slot": "bottom", "brand": "J.Crew"}, {"item": "white leather low-top sneakers", "slot": "shoes", "brand": "Common Projects"}, {"item": "tortoise wayfarer-style sunglasses", "slot": "accessory", "brand": "Persol"}]

Example output for "first day of law school":
[{"item": "navy unstructured wool blazer", "slot": "outerwear", "brand": "J.Crew"}, {"item": "light blue oxford button-down shirt", "slot": "top", "brand": "Brooks Brothers"}, {"item": "stone cotton chinos straight fit", "slot": "bottom", "brand": "Bonobos"}, {"item": "brown leather penny loafers", "slot": "shoes", "brand": "Allen Edmonds"}]
"""



def retrieve_context(query: str, n_results: int = TOP_K) -> tuple:
    """Retrieve relevant chunks from Pinecone. Returns (context_text, metadata_list)."""
    # Embed the query using Pinecone's inference API
    embed_result = _pc.inference.embed(
        model=EMBED_MODEL,
        inputs=[query],
        parameters={"input_type": "query"},
    )
    query_embedding = embed_result.data[0].values

    # Query Pinecone
    results = _pinecone_index.query(
        vector=query_embedding,
        top_k=n_results,
        include_metadata=True,
    )

    context_parts = []
    metadatas = []
    for match in results["matches"]:
        meta = match.get("metadata", {})
        doc = meta.get("text", "")
        title = meta.get("title", "Untitled")
        date = meta.get("date", "")
        context_parts.append(f'--- From "{title}" ({date}) ---\n{doc}')
        metadatas.append(meta)

    return "\n\n".join(context_parts), metadatas


def get_sources(metadatas):
    """Extract unique sources from metadata list."""
    sources = []
    seen_titles = set()
    for meta in metadatas:
        title = meta.get("title", "")
        if title and title not in seen_titles:
            seen_titles.add(title)
            sources.append({"title": title, "url": meta.get("url", ""), "date": meta.get("date", "")})
    return sources


def get_relevant_images(metadatas, max_images=6):
    """Get images from the blog posts that were referenced in the response."""
    images = []
    seen_titles = set()
    for meta in metadatas:
        title = meta.get("title", "")
        if title in seen_titles or title not in IMAGE_INDEX:
            continue
        seen_titles.add(title)
        post_images = IMAGE_INDEX[title]
        # Take up to 2 images per post
        for img in post_images[:2]:
            images.append({
                "file": img["file"],
                "src": f"/images/{img['file']}",
                "post_title": title,
            })
            if len(images) >= max_images:
                return images
    return images


def convert_image_for_claude(contents: bytes, content_type: str) -> tuple:
    """Convert any image to JPEG under 4MB for Claude API."""
    try:
        img = Image.open(io.BytesIO(contents))
        # Convert to RGB (handles HEIC, RGBA, palette modes, etc.)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Resize if very large (max 2048px on longest side)
        max_dim = 2048
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)

        # Save as JPEG with quality that keeps it under 4MB
        for quality in [85, 70, 50, 30]:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            if buf.tell() < 4 * 1024 * 1024:
                return buf.getvalue(), "image/jpeg"

        # If still too big, resize further
        img = img.resize((img.size[0] // 2, img.size[1] // 2), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=60)
        return buf.getvalue(), "image/jpeg"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {str(e)}")


def format_profile(profile) -> str:
    """Build a profile context string from a UserProfile."""
    if not profile:
        return ""
    parts = []
    if profile.height:
        parts.append(profile.height)
    if profile.build:
        parts.append(f"{profile.build} build")
    if profile.budget:
        budget_desc = {
            "budget": "value-conscious (prioritize best bang for the buck — affordable brands, sales, and smart buys)",
            "moderate": "willing to invest in quality pieces (mix of mid-range and premium where it matters most, like shoes and outerwear)",
            "luxury": "open to premium and luxury options (but still recommend the best product regardless of price — a $40 t-shirt can be better than a $200 one)",
        }
        desc = budget_desc.get(profile.budget, profile.budget)
        parts.append(desc)
    if not parts:
        return ""
    return "\n\nUser profile: " + ", ".join(parts) + ". Tailor your advice to their body type and budget preferences."


def ask_claude(system: str, user_content: list, max_tokens: int = 2048, temperature: float = 1.0) -> str:
    """Send a request to Claude and return the text response."""
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"Claude API error: {type(e).__name__}: {e}")
        raise


# ── FastAPI app ──────────────────────────────────────────────────────────────

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(
    title="Fashion Advisor",
    description="Menswear advice powered by Die, Workwear! and Claude",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
async def health_check():
    """Debug endpoint to check if all services are working."""
    status = {"server": "ok", "pinecone": "unknown", "anthropic": "unknown"}
    try:
        stats = _pinecone_index.describe_index_stats()
        status["pinecone"] = "ok"
        status["vectors"] = stats.get("total_vector_count", 0)
    except Exception as e:
        status["pinecone"] = f"error: {str(e)}"
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    status["anthropic"] = f"key set ({len(key)} chars)" if key else "NO KEY SET"
    return status


class UserProfile(BaseModel):
    height: Optional[str] = None
    build: Optional[str] = None
    budget: Optional[str] = None


class AdviceRequest(BaseModel):
    question: str
    context: Optional[str] = None
    profile: Optional[UserProfile] = None


class ShopRequest(BaseModel):
    need: str
    budget: Optional[str] = None
    context: Optional[str] = None


class AdviceResponse(BaseModel):
    answer: str


class VibeRequest(BaseModel):
    vibe: str
    profile: Optional[UserProfile] = None


class RefreshRequest(BaseModel):
    item: str
    brand: str = ""
    product: str = ""
    budget: str = ""
    exclude_links: List[str] = []


class ShopItem(BaseModel):
    name: str
    brand: str
    price: str
    link: str
    description: str
    image: str = ""
    search_item: str = ""
    search_product: str = ""
    slot: str = ""


class VibeResponse(BaseModel):
    vibe: str
    items: List[ShopItem]


import re as _re
import random as _random
from urllib.parse import urlparse as _urlparse

# Slot-generic fallback queries used by the final-guarantee block in /shop-vibe.
# When an essential slot (top/bottom/shoes) fails the brand-specific search AND
# the no-brand retry, we try these progressively-broader queries until one
# returns a real product. Without this the user can ship with a missing piece.
_FALLBACK_QUERIES = {
    "top": [
        "men's t-shirt",
        "men's cotton t-shirt",
        "men's button-down shirt",
        "men's athletic shirt",
    ],
    "bottom": [
        "men's chinos",
        "men's pants",
        "men's shorts",
        "men's jeans",
    ],
    "shoes": [
        "men's sneakers",
        "men's loafers",
        "men's leather dress shoes",
    ],
    "outerwear": [
        "men's jacket",
        "men's coat",
    ],
}

# ── Curated retailer list ─────────────────────────────────────────────────────
# Organized by price tier. search_product picks a random subset per query
# to ensure variety across recommendations.
# BRAND-DIRECT sites only — no multi-brand retailers like Mr Porter, SSENSE,
# Saks, End Clothing, Nordstrom, Farfetch, etc. Retailers are excluded
# because:
#   1. Variant-default og:image issues — they aggregate products from many
#      brands and their templates handle variants poorly (Saks served the
#      smooth penny loafer image for a pebble-grained product page).
#   2. Aggressive bot protection — most are behind Cloudflare/PerimeterX,
#      which blocks our image scraper and OOS detection.
#   3. Faceted-nav category leakage — their URL patterns produce category
#      pages that look like product pages to Google.
#   4. Lower affiliate margins than going direct to the brand.
# All multi-brand retailers are explicitly listed in `skip_domains` inside
# search_product so they're never returned even by brand-committed open-web
# searches.
RETAILERS = {
    "budget": [
        "uniqlo.com",
        "jcrew.com",
        "bananarepublic.com",
        "abercrombie.com",
        "cosstores.com",
        "uskees.com",
        "muttonheadstore.com",
    ],
    "mid": [
        "bonobos.com",
        "spiermackay.com",
        "percivalclo.com",
        "alexmill.com",
        "us.sandro-paris.com",
        "theory.com",
        "milworks.co",
        "heimat-textil.com",
        "deuscustoms.com",
        "mfpen.com",
    ],
    "premium": [
        "drakes.com",
        "sidmashburn.com",
        "jamesperse.com",
        "equipment.store",
        "warthog.vip",
        "beringia.world",
    ],
    "shoes": [
        "grantstoneshoes.com",
        "aldenshop.com",
        "paraboot.com",
        "ghbass.com",
        "meermin.com",
    ],
    "athletic": [
        "reigningchamp.com",
        "lululemon.com",
        "tenthousand.com",
        "outdoorvoices.com",
        "vuoriclothing.com",
        "rhone.com",
        "tracksmith.com",
        "satisfactionrunning.com",
        "nikeacg.com",
        "arcteryx.com",
        "cotopaxi.com",
        "on.com",
        "nobullproject.com",
        "setasports.com",
    ],
}

ALL_RETAILERS = []
for _tier in RETAILERS.values():
    ALL_RETAILERS.extend(_tier)


_ATHLETIC_KEYWORDS = {"gym", "workout", "running", "athletic", "athleisure",
                       "training", "exercise", "activewear", "jogger", "sneaker",
                       "performance", "sport", "hiking", "trail"}


def _is_athletic_item(item_text: str) -> bool:
    """Check if an item description suggests athletic/workout gear."""
    words = set(item_text.lower().split())
    return bool(words & _ATHLETIC_KEYWORDS)


def _get_site_query(budget: str = "", n: int = 8, item_hint: str = "") -> str:
    """Build a site: OR query from a random subset of retailers.
    Biases toward budget-friendly retailers when budget is 'budget',
    and toward athletic retailers for workout/athleisure items."""
    if _is_athletic_item(item_hint):
        # Athletic items: heavily favor athletic retailers, mix in some general
        pool = RETAILERS["athletic"] * 3 + RETAILERS["budget"] + RETAILERS["mid"]
    elif budget == "budget":
        pool = RETAILERS["budget"] * 2 + RETAILERS["mid"] + RETAILERS["shoes"]
    elif budget == "luxury":
        pool = RETAILERS["premium"] * 2 + RETAILERS["mid"] + RETAILERS["shoes"]
    else:
        pool = ALL_RETAILERS[:]

    subset = _random.sample(pool, min(n, len(pool)))
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in subset:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return " OR ".join(f"site:{s}" for s in unique)

def _extract_dollar_price(text: str) -> str:
    """Extract a clean dollar price from text like '$155.00', '£120', 'From $99', etc."""
    m = _re.search(r'\$\s*(\d[\d,]*(?:\.\d{2})?)', text)
    if m:
        raw = m.group(1).replace(",", "")
        try:
            val = float(raw)
            return f"${val:,.0f}" if val == int(val) else f"${val:,.2f}"
        except ValueError:
            pass
    return ""


_GENERIC_CATEGORY_TAILS = {
    # Generic category names that, when they're the LAST path segment,
    # indicate a browse page rather than a product page (e.g. SSENSE's
    # /en-pr/men/shoes).
    "shoes", "sneakers", "boots", "loafers", "sandals", "trainers",
    "shirts", "tshirts", "tees", "polos", "sweaters", "hoodies",
    "pants", "trousers", "jeans", "shorts", "chinos",
    "jackets", "coats", "blazers", "outerwear",
    "bags", "accessories", "hats", "belts", "sunglasses",
    "men", "mens", "women", "womens", "clothing", "apparel",
    "new", "new-arrivals", "sale", "all",
}


def _is_category_url(url: str) -> bool:
    """Check if a URL looks like a category/collection page rather than a product page."""
    lower = url.lower()
    category_signals = (
        "/collections/", "/collection/", "/category/", "/categories/",
        "/shop/", "/search", "/browse/", "/c/", "/plp/",
        "/buy/", "/mens-", "/womens-", "/all-",
        "?q=", "?query=", "?search=", "?sort=", "?sizes=", "?filter=",
        "?page=",
    )
    if any(s in lower for s in category_signals):
        return True
    # URL whose last path segment is a generic category name (no product slug
    # after) — catches SSENSE's /en-pr/men/shoes pattern, Mr Porter's
    # /us/mens/clothing/shoes, and similar.
    try:
        path = _urlparse(url).path.rstrip("/")
    except Exception:
        path = ""
    if path:
        last = path.rsplit("/", 1)[-1]
        if last in _GENERIC_CATEGORY_TAILS:
            return True
        # Inditex-group sites (Massimo Dutti, Zara, Stradivarius, Pull&Bear)
        # use `<category>-c<digits>` for category pages and `<product>-l<digits>`
        # / `<product>-p<digits>` for products. The "-c<num>" suffix is a
        # category indicator.
        if _re.search(r"-c\d+$", last):
            return True
    return False


def _is_non_us_locale(url: str) -> bool:
    """Catch URL paths that explicitly target a non-US locale, like
    SSENSE's /en-pr/ (Puerto Rico) or Mr Porter's /uk/ — these usually
    price in a non-USD currency and ship from elsewhere."""
    lower = url.lower()
    # /en-XX/ pattern where XX is not 'us'
    if _re.search(r"/en-(?!us)[a-z]{2}/", lower):
        return True
    # Other locale prefixes
    bad = ("/uk/", "/au/", "/ca/", "/nz/", "/ie/", "/sg/", "/hk/",
           "/in/", "/ae/", "/kr/", "/jp/", "/eu/", "/de/", "/fr/",
           "/it/", "/es/", "/nl/", "/be/", "/ch/", "/at/", "/pl/",
           "/dk/", "/se/", "/no/", "/fi/", "/br/", "/mx/")
    return any(b in lower for b in bad)


def _is_bad_title(title: str) -> bool:
    """Check if a search result title suggests a non-product or category page."""
    lower = title.lower()
    # Editorial / review / comparison / roundup pages. "for 2024" / "for 2025"
    # is a near-perfect signal of an editorial roundup ("Best Navy Sweaters
    # For 2025", "Our Favorite Boots For 2024"). "our favorite", "best of",
    # "must-have", "essential" are all roundup giveaways.
    bad_signals = (
        " vs ", " versus ", "comparison", "review:", "best ",
        "top 10", "top 5", "how to", "guide", "slim vs",
        "our favorite", "our favorites", "favorite", "favourites",
        "best of", "must-have", "must have", "essential ",
        "the best", "round-up", "roundup", "we love",
    )
    if any(s in lower for s in bad_signals):
        return True
    # Year-suffix pattern: "...For 2024", "...In 2025", "...Of 2026", etc.
    # Almost always an editorial article, not a product page.
    if _re.search(r"\b(?:for|in|of)\s+20\d{2}\b", lower):
        return True
    # Standalone year at the end: "...Sweaters 2025"
    if _re.search(r"\s20\d{2}$", lower) or _re.search(r"\s20\d{2}\b", lower):
        # Be careful not to false-positive on real product names that
        # contain a year (e.g. New Balance 2002R has "2002" in the name).
        # Only fire if the year is 2020+ (current/recent editorial era).
        m = _re.search(r"\b(20[2-9]\d)\b", lower)
        if m and int(m.group(1)) >= 2020 and int(m.group(1)) <= 2030:
            return True
    # Category / department page titles. SerpAPI titles for these are
    # almost always shaped like "Men's [category] - [Brand]" or
    # "[Brand] Men's [category]" or "Shop Men's [category]" with no
    # specific product descriptor (no fabric, color, fit, model).
    # Examples we've seen ship to users:
    #   "Men's Blazers - Massimo Dutti - US"
    #   "Men's Shirts | J.Crew"
    #   "Linen Pants for Mens"  (a search-results page)
    cat_words = (
        "blazers", "shirts", "pants", "trousers", "jeans", "shorts",
        "sweaters", "hoodies", "tops", "tees", "t-shirts", "polos",
        "jackets", "coats", "outerwear", "shoes", "sneakers", "boots",
        "loafers", "sandals", "bags", "accessories", "hats", "belts",
        "sunglasses", "underwear", "socks",
    )
    cat_prefix = ("men's ", "mens ", "shop men's ", "shop mens ",
                  "shop ", "men's clothing", "mens clothing")
    # Generic category title — e.g. "Men's Blazers - Massimo Dutti"
    for word in cat_words:
        for prefix in cat_prefix:
            if lower.startswith(prefix + word):
                return True
        # "[Category] for Men/Mens" pattern at start
        if lower.startswith(word + " for men") or lower.startswith(word + " for mens"):
            return True
    # "... for Mens" (plural) anywhere is a strong category SEO signal
    # used by content-farm product-listing pages ("Linen Pants for Mens").
    # Real product titles use "for Men" (singular).
    if " for mens" in lower:
        return True
    return False


# Minimum realistic dollar prices by slot. Anything cheaper is almost
# certainly a bad price extraction (snippet from a sale banner, "from $X"
# starting price, currency confusion). The accessory floor is lower because
# a $10 belt is plausible whereas a $5 pair of pants is not.
_MIN_PRICE_BY_SLOT = {
    "outerwear": 40,
    "top": 15,
    "bottom": 25,
    "shoes": 35,
    "bag": 20,
    "accessory": 8,
}


# Tokens used to detect Shopify's "variant-default og:image" trap, where
# the og:image points to a different product variant (e.g. the suede penny
# loafer URL serves the leather variant's og:image). We split them into
# fabric and color buckets so we can detect *contradictions* — if the slug
# names one fabric and the og:image filename names a different one, we
# know the og:image is wrong.
_FABRIC_TOKENS = {
    "suede", "leather", "wool", "linen", "cotton", "silk", "cashmere",
    "denim", "corduroy", "flannel", "oxford", "chambray", "twill",
    "gabardine", "fleece", "jersey", "knit", "mesh", "velvet", "satin",
    "velour", "canvas", "nylon", "polyester", "tweed", "shearling",
}
_COLOR_TOKENS = {
    "navy", "black", "white", "gray", "grey", "brown", "tan", "olive",
    "khaki", "beige", "cream", "ivory", "red", "blue", "green", "yellow",
    "pink", "purple", "orange", "burgundy", "maroon", "charcoal", "sand",
    "rust", "mint", "sage", "forest", "royal", "indigo", "ecru", "stone",
    "camel", "cognac", "oxblood", "midnight", "nautical",
}
_DESCRIPTIVE_TOKENS = _FABRIC_TOKENS | _COLOR_TOKENS


def _og_image_contradicts_slug(og_image_url: str, slug_tokens: set) -> bool:
    """Detect the Shopify variant-default og:image trap.

    Returns True if the og:image filename names a fabric or color that
    contradicts a fabric or color named in the URL slug. Doesn't reject
    just because a descriptor is missing from the filename — many real
    product images use abbreviated filenames. Only rejects when one
    side names a descriptor and the other side names a DIFFERENT
    descriptor in the same category.
    """
    if not og_image_url or not slug_tokens:
        return False
    fname = og_image_url.rsplit("/", 1)[-1].lower()
    fname_tokens = set(_re.findall(r"[a-z]{3,}", fname))

    slug_fabrics = slug_tokens & _FABRIC_TOKENS
    fname_fabrics = fname_tokens & _FABRIC_TOKENS
    if slug_fabrics and fname_fabrics and not (slug_fabrics & fname_fabrics):
        return True

    slug_colors = slug_tokens & _COLOR_TOKENS
    fname_colors = fname_tokens & _COLOR_TOKENS
    if slug_colors and fname_colors and not (slug_colors & fname_colors):
        return True

    return False


def _scrape_product_image(url: str, timeout: float = 5.0) -> str:
    """Try to extract the product image from the actual product page.
    Looks for og:image, twitter:image, or common product image meta tags.

    Returns "" if the product is out of stock (detected via schema.org JSON-LD).

    For pages where og:image points to a generic lookbook/campaign photo
    instead of the actual product (some Shopify stores do this — e.g.
    Save Khaki's twill short page sets og:image to a campaign shot, while
    the real product gallery lives under SKU-named filenames), fall through
    to gallery images whose filename overlaps with the URL slug.
    """
    try:
        resp = _requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        if resp.status_code != 200:
            return ""

        # Drop out-of-stock products via two signals:
        # 1. schema.org JSON-LD offers.availability — most reliable, used by
        #    Shopify and well-instrumented retailers. Bail if any offer says
        #    OutOfStock AND no offer says InStock (correctly handles variant
        #    products where some sizes are OOS but others are still available).
        # 2. Visible "sold out" / "out of stock" / "notify me" text in the
        #    page body — fallback for indie sites that don't emit structured
        #    data. We check for these phrases NEAR a "cart" button signal so
        #    we don't false-positive on review snippets that say things like
        #    "I keep selling out before I can get one."
        # IMPORTANT: scan the FULL HTML for both — availability markers and
        # OOS text are often deep in the page, not in the head.
        full_lower = resp.text.lower()
        if ("schema.org/outofstock" in full_lower
                and "schema.org/instock" not in full_lower):
            return ""
        # Text-based fallback. The phrases must appear AND there must be no
        # active "add to cart" / "add to bag" affordance for us to bail.
        oos_phrases = ("sold out", "out of stock", "notify me when",
                       "email me when available", "back in stock soon",
                       "currently unavailable")
        has_oos_text = any(p in full_lower for p in oos_phrases)
        # Active cart button signals → page is sellable
        has_active_cart = (
            'name="add"' in full_lower or "add-to-cart" in full_lower
            or "add to cart" in full_lower or "add to bag" in full_lower
        )
        # Disabled/hidden cart button → page is unsellable
        has_disabled_cart = (
            'disabled' in full_lower and ('add to cart' in full_lower
                                          or 'add to bag' in full_lower)
        )
        if has_oos_text and not has_active_cart:
            return ""
        if has_oos_text and has_disabled_cart:
            return ""

        head_html = resp.text[:100_000]   # head-only for og:image regex
        full_html = resp.text[:300_000]   # widened for gallery image scan

        # Step 1: try og:image first (most reliable for normal product pages).
        # Patterns use [^>]* rather than \s+ between <meta and attributes so
        # they match tags with extra attributes (e.g. React Helmet:
        # <meta data-react-helmet="true" property="og:image" content="...">).
        og_image = ""
        for pattern in [
            r'<meta[^>]*property=["\']og:image["\'][^>]*content=["\'](https?://[^"\']+)',
            r'<meta[^>]*content=["\'](https?://[^"\']+)["\'][^>]*property=["\']og:image',
            r'<meta[^>]*name=["\']twitter:image["\'][^>]*content=["\'](https?://[^"\']+)',
            r'<meta[^>]*content=["\'](https?://[^"\']+)["\'][^>]*name=["\']twitter:image',
        ]:
            m = _re.search(pattern, head_html, _re.IGNORECASE)
            if m:
                img_url = m.group(1)
                lower = img_url.lower()
                if not any(s in lower for s in ("logo", "favicon", "placeholder", "1x1", "pixel")):
                    og_image = img_url
                    break

        # Step 2: build slug tokens from the URL's last path segment, with
        # leading-zero variants. Used to detect whether og:image is actually
        # this product's image vs. a generic campaign shot.
        slug = _urlparse(url).path.rstrip("/").split("/")[-1].lower()
        slug_tokens = set()
        for t in _re.findall(r"[a-z0-9]{3,}", slug):
            slug_tokens.add(t)
            # "sk009071" -> also try "sk9071" (Shopify often drops zero pads)
            normed = _re.sub(r"(\D)0+(\d)", r"\1\2", t)
            if normed != t and len(normed) >= 3:
                slug_tokens.add(normed)

        def _has_slug_overlap(img_url: str) -> bool:
            fname = img_url.rsplit("/", 1)[-1].lower()
            return any(t in fname for t in slug_tokens)

        # Step 3: if og:image already references this product's slug, it's
        # the right image — return it without scanning the gallery.
        if og_image and slug_tokens and _has_slug_overlap(og_image):
            return og_image

        # Step 4: og:image is missing or generic. Hunt the gallery for an
        # image whose filename matches the URL slug. Take the first match
        # in source order so we get the "main" angle/color.
        if slug_tokens:
            for img in _re.findall(
                r'(https?://[^"\'\s>]+\.(?:jpg|jpeg|png|webp))',
                full_html,
                _re.IGNORECASE,
            ):
                lower = img.lower()
                if any(s in lower for s in (
                        "logo", "favicon", "placeholder", "1x1", "pixel", "icon", "sprite")):
                    continue
                if _has_slug_overlap(img):
                    return img

        # Step 5: fall back to og:image — but only if it doesn't contradict
        # the URL slug. Two distinct rejection paths:
        #
        # (a) Active contradiction. The slug says one fabric or color and
        #     the og:image filename says a different fabric or color in the
        #     same category — Shopify's variant-default trap (Todd Snyder's
        #     suede penny loafer page handing back the leather image).
        #
        # (b) No overlap when the slug names a specific variant. When the
        #     slug calls out a color/fabric AND the og:image filename has
        #     ZERO overlap with any slug token, we can't confirm the image
        #     matches the variant. This catches sites like Grant Stone,
        #     which use a single generic-named image (03SIDELoafers.jpg)
        #     for every color variant of the same shoe — the brown product
        #     photo gets served for the Crimson variant URL and the user
        #     sees the wrong shoe.
        if og_image and slug_tokens:
            if _og_image_contradicts_slug(og_image, slug_tokens):
                return ""
            if slug_tokens & _DESCRIPTIVE_TOKENS:
                fname = og_image.rsplit("/", 1)[-1].lower()
                if not any(t in fname for t in slug_tokens):
                    return ""
        return og_image
    except Exception:
        return ""


def search_product(item_info, budget="", exclude_links=None, force_no_brand=False) -> dict:
    """Search for a real, in-stock product using SerpAPI Google organic + images.

    When item_info contains a "brand" commitment AND force_no_brand is False,
    runs a brand-targeted open-web search ("men's <brand> <item>") so the
    result lands on the brand's actual product page. When no brand is given
    OR force_no_brand=True, falls back to the curated-retailer site filter
    (the original behavior). The retry chain in /shop-vibe uses
    force_no_brand=True to broaden a failed brand-committed search.
    """
    if exclude_links is None:
        exclude_links = []
    exclude_set = set(exclude_links)

    if isinstance(item_info, str):
        item, brand, product, slot = item_info, "", "", ""
    else:
        item = item_info.get("item", "")
        brand = item_info.get("brand", "") if not force_no_brand else ""
        product = item_info.get("product", "")
        slot = item_info.get("slot", "")

    # Build the search query.
    # - If Claude committed to a brand (and we're not forcing no-brand mode),
    #   open the search wide and target the brand directly. The brand is the
    #   precision signal; the site filter would only hurt.
    # - Otherwise, fall back to the original site-restricted curated-retailer
    #   pool with no brand bias.
    if brand:
        query = f"men's {brand} {item}"
        site_filter = ""
        serp_q = query
    else:
        query = f"men's {item}"
        site_filter = _get_site_query(budget, item_hint=item)
        serp_q = f"{query} ({site_filter})"

    skip_paths = ("/blog/", "/blogs/", "/article/", "/wiki/", "/news/",
                  "/review/", "/magazine/", "/editorial/", "/guide/")
    # Three categories of domains we never want returned, all in one tuple
    # so the search loop can short-circuit early:
    #
    # (1) Resale marketplaces — eBay, Poshmark, Grailed, etc. produce
    #     suspiciously cheap branded results ("$29.87 Buck Mason shorts").
    # (2) Social platforms / content farms — Pinterest, Reddit, YouTube,
    #     Wikipedia, etc. — never sell products themselves.
    # (3) Multi-brand retailers — Saks, Mr Porter, SSENSE, End Clothing,
    #     Nordstrom, Farfetch, etc. We deliberately go BRAND-DIRECT instead.
    #     Retailers cause variant-default og:image issues (Shopify-style
    #     templates serving the wrong variant's image), are aggressively
    #     bot-protected (Cloudflare/PerimeterX block our scraper), and pay
    #     lower affiliate margins than going direct to the brand.
    skip_domains = (
        # Resale & marketplaces
        "ebay.", "poshmark.com", "grailed.com", "etsy.com",
        "mercari.com", "depop.com", "vestiairecollective.",
        "therealreal.com", "stockx.com", "goat.com",
        "amazon.com/dp", "walmart.com",
        # Social / content farms
        "facebook.com", "instagram.com", "tiktok.com", "twitter.com",
        "x.com/", "pinterest.", "reddit.com", "youtube.com",
        "wikipedia.org",
        # Multi-brand retailers — go brand-direct instead
        "saksfifthavenue.com", "saksoff5th.com",
        "mrporter.com", "net-a-porter.com",
        "ssense.com",
        "endclothing.com", "end.clothing",
        "nordstrom.com", "nordstromrack.com",
        "neimanmarcus.com", "bergdorfgoodman.com", "bloomingdales.com",
        "macys.com",
        "farfetch.com", "matchesfashion.com", "mytheresa.com",
        "shopbop.com", "modaoperandi.com", "moda-operandi.com",
        "harrods.com", "selfridges.com", "liberty.co.uk",
        "24s.com", "yoox.com", "the-outnet.com", "gilt.com",
        "huckberry.com", "needsupply.com",
        "nomanwalksalone.com", "standardandstrange.com",
        "shoplostfound.com", "meridianboutique.com",
        "wittmore.com", "thereviveclub.com",
        "luisaviaroma.com", "browns.com", "brownsfashion.com",
        "stagprovisions.com",
        "zappos.com",  # multi-brand shoe marketplace
        # Brand-direct sites we've blocked because their template behavior
        # consistently causes wrong-image / wrong-variant issues that the
        # contradiction check can't fully fix.
        "toddsnyder.com",
        # Menswear content blogs / editorial roundup farms. These rank well
        # for product queries but link to articles, not product pages — and
        # the embedded affiliate links go through redirect chains we can't
        # follow cleanly. Always skip them in favor of going brand-direct.
        "gearmoose.com", "gearpatrol.com", "gearjunkie.com",
        "valetmag.com", "thecoolist.com", "thecoolector.com",
        "coolmaterial.com", "carryology.com", "iconmenstyle.com",
        "complex.com", "esquire.com", "gq.com", "menshealth.com",
        "highsnobiety.com", "hypebeast.com", "uncrate.com",
        "theadultman.com", "manofmany.com", "fashionbeans.com",
        "effortlessgent.com", "dappered.com", "permanentstyle.com",
        "putthison.com", "stylegirlfriend.com", "thevou.com",
        "menswearmusings.com", "thenobledandy.com",
        "hespokestyle.com", "opumo.com", "mensfashionmag.com",
        "businessinsider.com", "buzzfeed.com", "wirecutter.com",
        "nytimes.com/wirecutter",
    )

    result = {"query": query, "title": "", "link": "", "price": "", "image_url": ""}

    # Organic search for product link + price
    try:
        search = GoogleSearch({
            "engine": "google",
            "q": serp_q,
            "api_key": SERPAPI_KEY,
            "num": 15,
            "gl": "us",
        })
        data = search.get_dict()
        for r in data.get("organic_results", []):
            link = r.get("link", "")
            title = r.get("title", "")
            lower = link.lower()
            if link in exclude_set:
                continue
            if any(d in lower for d in skip_domains):
                continue
            if any(p in lower for p in skip_paths):
                continue
            if lower.rstrip("/").count("/") <= 2:
                continue
            if _is_category_url(lower):
                continue
            if _is_non_us_locale(lower):
                continue
            if _is_bad_title(title):
                continue
            # Filter out women's results (title or URL path signals).
            # NOTE: "women's" contains "men's" as a substring, so we must
            # strip the "women" tokens before checking for "men's" — a
            # naive `"men's" in title_lower` false-positives on women's
            # products and lets them through.
            title_lower = title.lower()
            if ("/women/" in lower or "/womens/" in lower or "/ladies/" in lower
                    or "women" in title_lower or "ladies" in title_lower):
                title_no_women = (
                    title_lower
                    .replace("women's", "")
                    .replace("womens", "")
                    .replace("women", "")
                )
                if ("men's" not in title_no_women
                        and "mens" not in title_no_women
                        and "for men" not in title_no_women):
                    continue

            # Extract price
            price = ""
            rich = r.get("rich_snippet") or {}
            bottom = rich.get("bottom") or {}
            exts = bottom.get("detected_extensions") or {}
            if exts.get("price"):
                price = f"${exts['price']:g}"
            elif exts.get("price_from"):
                price = f"${exts['price_from']:g}"
            else:
                # Fallback: parse first $XX from extensions list
                for ext_str in (bottom.get("extensions") or []):
                    extracted = _extract_dollar_price(ext_str)
                    if extracted:
                        price = extracted
                        break

            # Skip results without a clear dollar price
            if not price or not price.startswith("$"):
                continue
            # Sanity bounds on the extracted price. A $4 blazer or $5 pair
            # of pants is almost always a bad price extraction (a "from $X"
            # category snippet, a "save $X" sale banner, currency confusion).
            # The $20k upper bound catches foreign-currency leaks (Korean Won
            # jacket reads as ~$1.7M, Yen as ~$200k).
            try:
                numeric = float(price.replace("$", "").replace(",", ""))
                if numeric > 20000:
                    continue
                floor = _MIN_PRICE_BY_SLOT.get(slot, 5)
                if numeric < floor:
                    continue
            except ValueError:
                pass

            result["title"] = title
            result["link"] = link
            result["price"] = price
            break
    except Exception:
        pass

    # Image: try scraping OG image from the product page first (ensures match)
    if result["link"]:
        result["image_url"] = _scrape_product_image(result["link"])

        # Fallback: Google Images search if scraping failed
        if not result["image_url"]:
            try:
                img_query = f"{brand} {product or item}" if brand else item
                img_search = GoogleSearch({
                    "engine": "google_images",
                    "q": f"{img_query} product photo",
                    "api_key": SERPAPI_KEY,
                    "num": 5,
                })
                img_data = img_search.get_dict()
                for img in img_data.get("images_results", []):
                    img_url = img.get("original", "") or img.get("thumbnail", "")
                    if not img_url:
                        continue
                    img_lower = img_url.lower()
                    if any(d in img_lower for d in ("pinterest.", "reddit.", "youtube.", "wikimedia.")):
                        continue
                    result["image_url"] = img_url
                    break
            except Exception:
                pass

    return result


@app.get("/images/{filename}")
async def serve_image(filename: str):
    """Serve a scraped blog image."""
    path = os.path.join(IMAGES_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@app.post("/advice", response_model=AdviceResponse)
async def get_advice(req: AdviceRequest):
    """Get text-based fashion advice."""
    search_query = req.question
    if req.context:
        search_query += f" {req.context}"

    context, metadatas = retrieve_context(search_query)

    user_message = f"""Here are relevant excerpts from the Die, Workwear! blog to inform your advice:

{context}

---

User's question: {req.question}"""
    if req.context:
        user_message += f"\nAdditional context: {req.context}"
    user_message += format_profile(req.profile)

    answer = ask_claude(SYSTEM_PROMPT, [{"type": "text", "text": user_message}])

    return AdviceResponse(answer=answer)


@app.post("/outfit-check", response_model=AdviceResponse)
async def check_outfit(
    image: UploadFile = File(...),
    description: str = Form(""),
    profile: str = Form(""),
):
    """Upload an outfit photo for style critique, with optional text description."""
    contents = await image.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be under 20MB")

    # Convert to JPEG and compress for Claude API
    img_bytes, media_type = convert_image_for_claude(contents, image.content_type or "")
    b64_image = base64.b64encode(img_bytes).decode("utf-8")

    # Retrieve context about outfit evaluation
    search_terms = "outfit evaluation fit proportions style critique clothing"
    if description:
        search_terms = description + " " + search_terms
    context, metadatas = retrieve_context(search_terms)

    desc_line = ""
    if description.strip():
        desc_line = f"\n\nThe user describes their outfit as: {description.strip()}\n\nUse this description to correctly identify the pieces in the photo."

    # Parse profile from JSON string (multipart form can't send nested objects)
    profile_obj = None
    if profile:
        try:
            import json as _json
            pdata = _json.loads(profile)
            profile_obj = UserProfile(**pdata)
        except Exception:
            pass
    profile_line = format_profile(profile_obj)

    user_content = [
        {
            "type": "text",
            "text": f"""Here are relevant style insights from expert menswear sources:

{context}

---

The user has uploaded a photo of their outfit.{desc_line} Break down every piece in the outfit — what it is, how it fits, and how it works with the rest. Then give an honest, constructive overall critique with specific suggestions for improvement.

Important: If any item (especially footwear) is hard to make out due to distance, lighting, or angle, say so honestly rather than guessing. For example, say "these appear to be X, though it's hard to tell from the photo" instead of stating it as fact.{profile_line}""",
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64_image,
            },
        },
    ]

    answer = ask_claude(SYSTEM_PROMPT, user_content, max_tokens=3000, temperature=0.3)

    return AdviceResponse(answer=answer)


@app.post("/shop", response_model=AdviceResponse)
async def get_shopping_recs(req: ShopRequest):
    """Get shopping recommendations at various price points."""
    search_query = f"where to buy {req.need} shopping recommendations brands"
    if req.budget:
        search_query += f" {req.budget}"
    if req.context:
        search_query += f" {req.context}"

    context, metadatas = retrieve_context(search_query)

    user_message = f"""Here are relevant excerpts from the Die, Workwear! blog about shopping and brands:

{context}

---

The user is looking for shopping recommendations.
What they need: {req.need}"""
    if req.budget:
        user_message += f"\nBudget: {req.budget}"
    if req.context:
        user_message += f"\nAdditional context: {req.context}"

    user_message += """

Please provide specific brand and store recommendations organized by price tier (budget, mid-range, premium) where applicable. Include direct website links for each brand/store as markdown links [Name](url). Include both online and brick-and-mortar options when relevant."""

    answer = ask_claude(SHOP_SYSTEM_PROMPT, [{"type": "text", "text": user_message}])

    return AdviceResponse(answer=answer)


@app.post("/shop-vibe", response_model=VibeResponse)
async def get_vibe_recommendations(req: VibeRequest):
    """Get structured shopping recommendations based on a vibe/occasion."""
    # Phase 1: Claude determines what items are needed for the vibe
    search_query = f"outfit for {req.vibe} what to wear style"
    try:
        context, metadatas = retrieve_context(search_query)
    except Exception:
        context, metadatas = "", []

    items_message = f"""Here are relevant excerpts from the Die, Workwear! blog:

{context}

---

{CATALOG_PROMPT_TEXT}

---

The user wants to dress for this vibe/occasion: "{req.vibe}"
{format_profile(req.profile)}
Return a JSON array of 6-8 item descriptions for a complete outfit that nails this vibe. Each object should include "item", "slot", AND "brand" (commit to a brand from the curated catalog when one fits the vibe, otherwise return brand=""). Remember: ONLY output the JSON array, nothing else."""

    try:
        raw_items = ask_claude(VIBE_ITEMS_PROMPT, [{"type": "text", "text": items_message}])
    except Exception as e:
        logger.error(f"shop-vibe Claude call failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail=f"Style advisor is temporarily unavailable — {type(e).__name__}: {e}")

    try:
        cleaned = raw_items.strip()
        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        # Extract JSON array even if surrounded by text
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]
        item_descriptions = json.loads(cleaned)
    except (json.JSONDecodeError, IndexError, ValueError):
        raise HTTPException(status_code=502, detail="Could not build outfit — please try again")

    # Phase 2: Search for real, in-stock products for each item
    budget = req.profile.budget if req.profile else ""
    loop = asyncio.get_event_loop()
    search_tasks = [
        loop.run_in_executor(_executor, search_product, item_info, budget)
        for item_info in item_descriptions
    ]
    search_results = await asyncio.gather(*search_tasks)

    # Build ShopItems from successful search results, queue failed essentials
    items = []
    failed_essential = []  # essential slots (top/bottom/shoes) that need retry
    for item_info, sr in zip(item_descriptions, search_results):
        if isinstance(item_info, str):
            s_item = item_info
            slot = "accessory"
        else:
            s_item = item_info.get("item", "")
            slot = item_info.get("slot", "accessory")

        if not sr["link"] or not sr.get("price") or not sr.get("image_url"):
            if slot in ("top", "bottom", "shoes"):
                failed_essential.append(item_info)
            continue

        items.append(ShopItem(
            name=sr["title"],
            brand="",
            price=sr["price"],
            link=sr["link"],
            description=s_item,
            image=sr["image_url"],
            search_item=s_item,
            search_product="",
            slot=slot,
        ))

    # Retry chain for failed essential slots.
    # Tier 1: re-search with brand stripped (force_no_brand=True). Claude's
    # first brand commitment may have been out of stock or 404'd; falling
    # back to the curated-retailer site filter often finds a similar item.
    if failed_essential:
        retry_tasks = [
            loop.run_in_executor(
                _executor, search_product, item_info, budget, None, True
            )
            for item_info in failed_essential
        ]
        retry_results = await asyncio.gather(*retry_tasks)
        still_failed = []
        for item_info, sr in zip(failed_essential, retry_results):
            if not sr["link"] or not sr.get("price") or not sr.get("image_url"):
                still_failed.append(item_info)
                continue
            s_item = item_info if isinstance(item_info, str) else item_info.get("item", "")
            slot = "accessory" if isinstance(item_info, str) else item_info.get("slot", "accessory")
            items.append(ShopItem(
                name=sr["title"], brand="", price=sr["price"],
                link=sr["link"], description=s_item, image=sr["image_url"],
                search_item=s_item, search_product="", slot=slot,
            ))
        failed_essential = still_failed

    # Tier 2 (final guarantee): for any essential slot STILL missing — either
    # because Claude omitted it from JSON entirely OR because all branded +
    # broadened searches failed — try a list of progressively broader
    # slot-generic queries until one returns a real product. Without this
    # the user can ship with no shirt for an "outdoor concert" vibe.
    delivered_slots = {it.slot for it in items}
    for required_slot in ("top", "bottom", "shoes"):
        if required_slot in delivered_slots:
            continue
        for fq in _FALLBACK_QUERIES.get(required_slot, []):
            sr = await loop.run_in_executor(
                _executor, search_product,
                {"item": fq, "brand": "", "product": ""}, budget, None, True
            )
            if sr["link"] and sr.get("price") and sr.get("image_url"):
                items.append(ShopItem(
                    name=sr["title"], brand="", price=sr["price"],
                    link=sr["link"], description=fq, image=sr["image_url"],
                    search_item=fq, search_product="", slot=required_slot,
                ))
                logger.info(
                    "shop-vibe '%s': final-guarantee filled '%s' with '%s'",
                    req.vibe, required_slot, fq,
                )
                break
        else:
            logger.error(
                "shop-vibe '%s': FINAL-GUARANTEE FAILED for '%s' — "
                "essential slot will be missing",
                req.vibe, required_slot,
            )

    return VibeResponse(vibe=req.vibe, items=items)


@app.post("/shop-refresh", response_model=ShopItem)
async def refresh_product(req: RefreshRequest):
    """Get an alternative product for the same item category, excluding previously seen links."""
    item_info = {"item": req.item}
    loop = asyncio.get_event_loop()
    sr = await loop.run_in_executor(
        _executor, search_product, item_info, req.budget, req.exclude_links
    )
    if not sr["link"] or not sr.get("price") or not sr.get("image_url"):
        raise HTTPException(status_code=404, detail="No more alternatives found")
    return ShopItem(
        name=sr["title"],
        brand="",
        price=sr["price"],
        link=sr["link"],
        description=req.item,
        image=sr["image_url"],
    )


