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
HAIKU_MODEL = "claude-haiku-4-5-20251001"
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

You MUST respond with ONLY a valid JSON array of 5-7 objects. No markdown, no explanation, no preamble — just the JSON array.

Each object must have:
- "item": a specific, searchable garment or accessory description (style, color, fabric, fit). Be specific enough that searching for it on a retailer's site would find the right kind of product.
- "slot": one of "outerwear", "top", "bottom", "shoes", or "accessory" — indicates what part of the outfit this is.

SLOT RULES — follow these strictly:
- "outerwear": jackets, blazers, coats, vests, overshirts. Skip if not needed for the vibe.
- "top": shirts, t-shirts, polos, sweaters, henleys — anything worn on the upper body as the main layer.
- "bottom": trousers, pants, jeans, chinos, shorts.
- "shoes": any footwear — sneakers, loafers, boots, dress shoes.
- "accessory": ONLY small add-ons like belts, watches, sunglasses, ties, pocket squares, hats, scarves, socks. A polo, sweater, or knit shirt is NEVER an accessory — those are "top".

Do NOT include brand names — the system will search across a curated set of retailers automatically. Focus purely on describing the right ITEMS for the vibe.

Cover a full outfit: outerwear (if needed), exactly 1 top, exactly 1 bottom, exactly 1 pair of shoes, and 1-2 accessories.

Think carefully about what the occasion ACTUALLY calls for:
- Boat day / beach / pool → swim trunks as the bottom, skip belts, skip outerwear
- Gym / workout → athletic shorts or joggers, performance top, trainers
- Black tie → tuxedo pieces, dress shoes, bow tie
- Casual → don't force a belt or tie if they don't add to the look
Only recommend accessories that genuinely complete the outfit for that specific scenario.

Example output:
[{"item": "unstructured navy linen blazer", "slot": "outerwear"}, {"item": "white linen spread-collar shirt", "slot": "top"}, {"item": "cream cotton chinos slim fit", "slot": "bottom"}, {"item": "brown leather penny loafers", "slot": "shoes"}, {"item": "navy knit silk tie", "slot": "accessory"}, {"item": "white linen pocket square", "slot": "accessory"}]

For every item, also add a "reason" field: a ≤12-word explanation of why this specific choice fits the vibe. Reference concrete style logic (fabric, color, silhouette, climate). No filler words like "versatile" or "stylish".
"""

INTENT_CLASSIFIER_PROMPT = """You classify men's fashion shopping queries into one of four modes.

- "piece": user wants a single category of item with multiple options to choose from. Examples: "loafers under $400 for Miami summer", "warm coats for NYC winter", "linen shirts for the beach", "black jeans that drape well".
- "outfit": user wants a full head-to-toe look for a vibe or occasion. Examples: "smart casual Friday", "beach wedding in Positano", "first date at a cocktail bar", "ranch wedding in Texas".
- "exploratory": user wants fresh ideas outside their usual. Examples: "show me something unexpected for fall", "surprise me", "break me out of my rut", "something I wouldn't normally wear".
- "clarify": query is too vague to act on without more info. Examples: "clothes", "help", "something nice", "idk".

Respond with ONLY one word: piece, outfit, exploratory, or clarify. No explanation, no punctuation."""

PIECE_ITEMS_PROMPT = SYSTEM_PROMPT + """
You are helping a user shop for a specific TYPE of item. They've asked for a single category (shoes, shirts, jackets, etc.) and want to see multiple strong options.

You MUST respond with ONLY a valid JSON array of 4-6 objects. No markdown, no explanation, no preamble — just the JSON array.

Each object must have:
- "item": a specific, searchable garment description (style, color, material, fit). Each item should be a GENUINELY different take on the user's request — different materials, colors, silhouettes, or levels of formality. Not slight tweaks of the same thing.
- "reason": a ≤12-word explanation of why this specific pick fits their query. Reference concrete style logic (climate, fabric, silhouette, occasion). No filler ("versatile", "stylish", "classic choice").

Do NOT include brand names — the system searches curated retailers automatically.

Example for "loafers under $400 for Miami summer":
[
  {"item": "brown suede tassel loafer", "reason": "Summer-weight suede breathes in Miami humidity"},
  {"item": "navy woven leather loafer", "reason": "Open weave lets heat escape on hot days"},
  {"item": "cream canvas penny loafer", "reason": "Light color and casual fabric suit resort settings"},
  {"item": "burgundy horsebit loafer", "reason": "Warm tone adds depth for evening wear"},
  {"item": "olive suede bit loafer", "reason": "Earthy color pairs with linen and natural fabrics"}
]
"""

EXPLORATORY_ITEMS_PROMPT = SYSTEM_PROMPT + """
The user wants FRESH ideas they wouldn't normally think of — directional picks that broaden their taste without abandoning it.

You MUST respond with ONLY a valid JSON array of 3-5 objects. No markdown, no explanation — just JSON.

Each object must have:
- "item": a specific, searchable garment description. Push harder than "safe". Reach for a texture, silhouette, or color they might not own yet.
- "reason": a ≤12-word explanation of why this is worth trying. Reference a concrete aesthetic or use-case, not filler.

Do NOT include brand names. Keep items genuinely wearable — directional but not costumey.

Example:
[
  {"item": "slouchy pleated wool trouser in cream", "reason": "Replaces chinos with more drape and winter warmth"},
  {"item": "shawl-collar chocolate cardigan", "reason": "Softer alternative to a blazer for casual dinners"},
  {"item": "brown suede chelsea boot with crepe sole", "reason": "Grippier, softer take on a classic silhouette"}
]
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


def ask_claude(system: str, user_content: list, max_tokens: int = 2048, temperature: float = 1.0, model: str = CLAUDE_MODEL) -> str:
    """Send a request to Claude and return the text response."""
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"Claude API error: {type(e).__name__}: {e}")
        raise


def classify_intent(query: str) -> str:
    """Classify a user query into piece, outfit, exploratory, or clarify.
    Uses Haiku for speed and cost."""
    try:
        result = ask_claude(
            INTENT_CLASSIFIER_PROMPT,
            [{"type": "text", "text": query}],
            max_tokens=10,
            temperature=0.0,
            model=HAIKU_MODEL,
        )
        mode = result.strip().lower().rstrip(".,!?")
        if mode in ("piece", "outfit", "exploratory", "clarify"):
            return mode
    except Exception as e:
        logger.error(f"classify_intent failed: {e}")
    return "outfit"  # safe default


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
    shoeSize: Optional[str] = None


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


class QueryRequest(BaseModel):
    query: str
    profile: Optional[UserProfile] = None


class RefreshRequest(BaseModel):
    item: str
    brand: str = ""
    product: str = ""
    budget: str = ""
    exclude_links: List[str] = []
    max_price: Optional[int] = None


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
    reason: str = ""  # 12-word "why this for you" explanation


class VibeResponse(BaseModel):
    vibe: str
    items: List[ShopItem]


class QueryResponse(BaseModel):
    mode: str  # "piece", "outfit", "exploratory", or "clarify"
    query: str
    items: List[ShopItem] = []
    clarify: str = ""  # follow-up question shown when mode == "clarify"


import re as _re
import random as _random

# ── Curated retailer list ─────────────────────────────────────────────────────
# Organized by price tier. search_product picks a random subset per query
# to ensure variety across recommendations.
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
        "toddsnyder.com",
        "bonobos.com",
        "spiermackay.com",
        "percivalclo.com",
        "alexmill.com",
        "us.sandro-paris.com",
        "theory.com",
        "milworks.co",
        "heimat-textil.com",
        "wittmore.com",
        "deuscustoms.com",
        "thereviveclub.com",
        "mfpen.com",
    ],
    "premium": [
        "drakes.com",
        "sidmashburn.com",
        "jamesperse.com",
        "mrporter.com",
        "nomanwalksalone.com",
        "bergbergstore.com",
        "saksfifthavenue.com",
        "endclothing.com",
        "ssense.com",
        "standardandstrange.com",
        "shoplostfound.com",
        "meridianboutique.com",
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


def _price_to_float(price_str: str) -> float:
    """'$1,234.56' → 1234.56. Returns 0.0 if parse fails."""
    if not price_str:
        return 0.0
    m = _re.search(r'(\d[\d,]*(?:\.\d{1,2})?)', price_str.replace("$", ""))
    if not m:
        return 0.0
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return 0.0


def _extract_max_price(text: str) -> Optional[int]:
    """Parse a max price cap from a natural-language query.
    Handles 'under $400', 'below $1,200', 'less than $300', 'sub-$500', 'max $250',
    '$400 or less', 'up to $500', 'budget $300'. Returns None if no cap found."""
    if not text:
        return None
    t = text.lower().replace(",", "")
    patterns = [
        r'(?:under|below|less than|max|maximum|up to|sub[- ]?|budget(?: of)?)\s*\$?\s*(\d+)',
        r'\$?\s*(\d+)\s*(?:or less|or under|max|maximum)',
    ]
    for pat in patterns:
        m = _re.search(pat, t)
        if m:
            try:
                val = int(m.group(1))
                if 20 <= val <= 20000:  # sane bounds
                    return val
            except ValueError:
                continue
    return None


CATEGORY_WORDS = {
    "caps", "hats", "shirts", "pants", "jeans", "shoes", "boots",
    "sneakers", "loafers", "blazers", "jackets", "coats", "belts",
    "ties", "shorts", "sweaters", "polos", "tees", "accessories",
    "outerwear", "knitwear", "footwear", "bags", "sale", "new",
    "trousers", "chinos", "shirting", "tailoring", "denim",
    "elevated-casual", "smart-casual", "casual", "formal", "workwear",
    "tops", "bottoms", "swimwear", "underwear", "socks", "hosiery",
    "clothing", "apparel", "mens", "men", "womens", "women",
    "sandals", "slippers", "slides", "mules", "oxfords", "derbies",
    "brogues", "trainers", "flats", "heels", "eyewear", "sunglasses",
    "watches", "wallets", "scarves", "gloves", "ready-to-wear",
}

MATERIAL_WORDS = {
    "suede", "leather", "canvas", "cotton", "wool", "linen", "silk",
    "nylon", "denim", "cashmere", "velvet", "satin", "tweed", "corduroy",
}


def _is_women_only(url: str, title: str) -> bool:
    """Check if a URL or title points to women's-only content."""
    lower_url = url.lower()
    lower_title = title.lower()
    # URL path segments that mark women's sections
    path = lower_url.split("?", 1)[0]
    if any(s in path for s in ("/women/", "/womens/", "/women-", "/ladies/", "/ladies-")):
        return True
    # Title contains women's with no men's counterbalance
    has_womens = "women" in lower_title or "ladies" in lower_title
    has_mens = "for men" in lower_title or "men's" in lower_title or "mens" in lower_title
    if has_womens and not has_mens:
        return True
    return False


def _is_category_url(url: str) -> bool:
    """Check if a URL looks like a category/collection/brand-landing page rather than a product page.

    Strategy: flag URLs whose STRUCTURE says 'listing' (specific path fragments
    known to host categories), or whose trailing segment is a generic word."""
    lower = url.lower()
    # Query-param listing signals (search pages)
    if any(s in lower for s in ("?q=", "?query=", "?search=", "?filter=")):
        return True
    # Strip query string before path analysis (srsltid=... etc. is noise).
    path = lower.split("?", 1)[0]
    segments = [s for s in path.rstrip("/").split("/") if s]

    # Hard substring signals — path fragments ONLY used on listing pages.
    hard_signals = (
        "/collections/", "/collection/", "/category/", "/categories/",
        "/shop-by/", "/shop_by/", "/shopby/",
        "/browse/", "/plp/", "/department/", "/dept/", "/catalog/",
        "/all-", "/search",
        # Saks uses /c/ for categories and /brand/ for brand landing pages
        "/c/", "/brand/", "/brands/",
        # Mr Porter uses /designer/ for designer category listings
        # (product URLs use /product/ instead)
        "/designer/", "/designers/",
    )
    if any(s in path for s in hard_signals):
        return True

    # URL ends at a generic category word → listing page
    # (e.g. mrporter.com/en-us/mens/clothing/shoes/loafers)
    if segments:
        last = segments[-1].split(".")[0]  # strip .html etc.

        # Direct category word
        if last in CATEGORY_WORDS:
            return True

        # Compound category slug like "suede-shoes", "dress-shoes",
        # "slides-sandals", "leather-loafers". If EVERY part of the hyphenated
        # slug is a category or material word, it's a listing.
        if "-" in last:
            parts = last.split("-")
            if len(parts) <= 4 and all(
                p in CATEGORY_WORDS or p in MATERIAL_WORDS for p in parts
            ):
                return True
            # Also flag if the LAST hyphen-word is a category AND the slug is short.
            # (Catches "summer-shoes" where "summer" isn't in either set.)
            if parts[-1] in CATEGORY_WORDS and len(parts) <= 3:
                return True

        # Very short trailing segment with no digits — suspect category filter
        # (e.g. /c/men/shoes/slides-sandals/white where last is a color filter)
        if len(last) <= 6 and last.isalpha() and len(segments) >= 3:
            return True
    return False


def _is_bad_title(title: str) -> bool:
    """Check if a search result title suggests a non-product or category page."""
    lower = title.lower()
    bad_signals = (
        " vs ", " versus ", "comparison", "review:", "best ",
        "top 10", "top 5", "how to", "guide", "slim vs",
        "shop men", "shop women", "shop all", "browse ",
        "| all ", "collection |", "collections |",
    )
    if any(s in lower for s in bad_signals):
        return True

    # Patterns like "Shoes for Men", "Suede Shoes for Men", "Loafers for Women"
    # These are always category pages, never single products.
    if _re.search(r'\b\w+s\s+for\s+(men|women)\b', lower):
        return True

    # Category pages often have titles like "Men's Elevated Casual | END. (US)"
    # or "Loafers | Mr Porter" — pure category word before a pipe.
    if "|" in lower:
        pre = lower.split("|", 1)[0].strip()
        pre_words = pre.replace("'s", "").replace("-", " ").split()
        if 1 <= len(pre_words) <= 4:
            if any(w in CATEGORY_WORDS for w in pre_words):
                return True
    return False


def _scrape_product_image(url: str, timeout: float = 5.0) -> str:
    """Try to extract the product image from the actual product page.
    Looks for og:image, twitter:image, or common product image meta tags."""
    try:
        resp = _requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        if resp.status_code != 200:
            return ""
        html = resp.text[:100_000]  # Only scan first 100KB

        # Try og:image first (most reliable for product pages)
        for pattern in [
            r'<meta\s+property=["\']og:image["\']\s+content=["\'](https?://[^"\']+)["\']',
            r'<meta\s+content=["\'](https?://[^"\']+)["\']\s+property=["\']og:image["\']',
            r'<meta\s+name=["\']twitter:image["\']\s+content=["\'](https?://[^"\']+)["\']',
            r'<meta\s+content=["\'](https?://[^"\']+)["\']\s+name=["\']twitter:image["\']',
        ]:
            m = _re.search(pattern, html, _re.IGNORECASE)
            if m:
                img_url = m.group(1)
                # Skip placeholder/logo images
                lower = img_url.lower()
                if any(s in lower for s in ("logo", "favicon", "placeholder", "1x1", "pixel")):
                    continue
                return img_url
        return ""
    except Exception:
        return ""


def search_product(item_info, budget="", exclude_links=None, shoe_size="", max_price: Optional[int] = None) -> dict:
    """Search for a real, in-stock product using SerpAPI Google organic + images.

    max_price (if set) is a hard ceiling in USD — results priced above it are skipped."""
    if exclude_links is None:
        exclude_links = []
    exclude_set = set(exclude_links)

    if isinstance(item_info, str):
        item, brand, product, slot = item_info, "", "", ""
    else:
        item = item_info.get("item", "")
        brand = item_info.get("brand", "")
        product = item_info.get("product", "")
        slot = item_info.get("slot", "")

    # Build item query (no brand bias — let the site: filter handle sourcing)
    query = f"men's {item}"
    # Append shoe size for footwear queries to favor in-stock results
    if shoe_size and slot == "shoes":
        query += f" size {shoe_size}"
    # Hint the search engine at the price range — helps bias toward in-budget results
    if max_price:
        query += f" under ${max_price}"

    # Build site-restricted search query (pass item text for athletic detection)
    site_filter = _get_site_query(budget, item_hint=item)

    skip_paths = ("/blog/", "/blogs/", "/article/", "/wiki/", "/news/",
                  "/review/", "/magazine/", "/editorial/", "/guide/")

    result = {"query": query, "title": "", "link": "", "price": "", "image_url": ""}

    # Organic search for product link + price
    try:
        search = GoogleSearch({
            "engine": "google",
            "q": f"{query} ({site_filter})",
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
            if any(p in lower for p in skip_paths):
                continue
            if lower.rstrip("/").count("/") <= 2:
                continue
            if _is_category_url(lower):
                continue
            if _is_bad_title(title):
                continue
            if _is_women_only(lower, title):
                continue

            # Extract price
            price = ""
            rich = r.get("rich_snippet") or {}
            bottom = rich.get("bottom") or {}
            exts = bottom.get("detected_extensions") or {}
            if exts.get("price"):
                p = exts['price']
                price = f"${p:,.0f}" if p == int(p) else f"${p:,.2f}"
            elif exts.get("price_from"):
                p = exts['price_from']
                price = f"${p:,.0f}" if p == int(p) else f"${p:,.2f}"
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

            # Enforce hard price ceiling — small slack (5%) to absorb rounding/display noise
            if max_price:
                val = _price_to_float(price)
                if val > max_price * 1.05:
                    continue

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


def _parse_items_json(raw: str) -> list:
    """Extract and parse a JSON array from Claude's response, tolerant of fences/preamble."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0]
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1:
        cleaned = cleaned[start:end + 1]
    return json.loads(cleaned)


def _item_to_shopitem(item_info, sr) -> ShopItem:
    """Build a ShopItem from a Claude item description + search result."""
    if isinstance(item_info, str):
        s_item, slot, reason = item_info, "", ""
    else:
        s_item = item_info.get("item", "")
        slot = item_info.get("slot", "")
        reason = item_info.get("reason", "")
    return ShopItem(
        name=sr["title"],
        brand="",
        price=sr["price"],
        link=sr["link"],
        description=s_item,
        image=sr["image_url"],
        search_item=s_item,
        search_product="",
        slot=slot,
        reason=reason,
    )


async def _search_and_build(item_descriptions: list, profile: Optional[UserProfile], retry_slots=(), max_price: Optional[int] = None) -> List[ShopItem]:
    """Run parallel product searches for a list of item descriptions and return ShopItems.
    If retry_slots is provided, failed searches for items in those slots get retried once.
    max_price (optional) caps per-item price — results above it are skipped."""
    budget = profile.budget if profile else ""
    shoe_size = profile.shoeSize if profile else ""
    loop = asyncio.get_event_loop()
    search_tasks = [
        loop.run_in_executor(_executor, search_product, item_info, budget, None, shoe_size, max_price)
        for item_info in item_descriptions
    ]
    search_results = await asyncio.gather(*search_tasks)

    items = []
    failed_retry = []
    for item_info, sr in zip(item_descriptions, search_results):
        if not sr["link"] or not sr.get("price") or not sr.get("image_url"):
            slot = "" if isinstance(item_info, str) else item_info.get("slot", "")
            if slot in retry_slots:
                failed_retry.append(item_info)
            continue
        items.append(_item_to_shopitem(item_info, sr))

    if failed_retry:
        retry_tasks = [
            loop.run_in_executor(_executor, search_product, item_info, budget, None, shoe_size, max_price)
            for item_info in failed_retry
        ]
        retry_results = await asyncio.gather(*retry_tasks)
        for item_info, sr in zip(failed_retry, retry_results):
            if not sr["link"] or not sr.get("price") or not sr.get("image_url"):
                continue
            items.append(_item_to_shopitem(item_info, sr))

    return items


async def _generate_outfit(query: str, profile: Optional[UserProfile]) -> List[ShopItem]:
    """Generate a full outfit for a vibe/occasion query."""
    max_price = _extract_max_price(query)
    try:
        context, _ = retrieve_context(f"outfit for {query} what to wear style")
    except Exception:
        context = ""

    items_message = f"""Here are relevant excerpts from expert menswear sources:

{context}

---

The user wants to dress for this vibe/occasion: "{query}"
{format_profile(profile)}
Return a JSON array of 5-7 item descriptions for a complete outfit that nails this vibe. Remember: ONLY output the JSON array, nothing else."""

    try:
        raw_items = ask_claude(VIBE_ITEMS_PROMPT, [{"type": "text", "text": items_message}])
    except Exception as e:
        logger.error(f"outfit generation failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail=f"Style advisor is temporarily unavailable — {type(e).__name__}: {e}")

    try:
        item_descriptions = _parse_items_json(raw_items)
    except (json.JSONDecodeError, IndexError, ValueError):
        raise HTTPException(status_code=502, detail="Could not build outfit — please try again")

    return await _search_and_build(item_descriptions, profile, retry_slots=("top", "bottom", "shoes"), max_price=max_price)


async def _generate_piece(query: str, profile: Optional[UserProfile]) -> List[ShopItem]:
    """Generate 4-6 variations of a single item category."""
    max_price = _extract_max_price(query)
    try:
        context, _ = retrieve_context(query)
    except Exception:
        context = ""

    budget_line = f"\nHARD PRICE CEILING: ${max_price}. Every pick must realistically exist under this price — no aspirational picks above it." if max_price else ""

    items_message = f"""Here are relevant excerpts from expert menswear sources:

{context}

---

The user is shopping for: "{query}"
{format_profile(profile)}{budget_line}

CRITICAL: Every variation MUST fit the full context of the query — climate, season, setting, formality. For "Miami summer", every pick must be summer-weight and warm-climate appropriate. No black calf city shoes, no wool, no weather-resistant finishes unless the query asks for it.

Return a JSON array of 4-6 item variations that give them genuine choice. Remember: ONLY output the JSON array."""

    try:
        raw_items = ask_claude(PIECE_ITEMS_PROMPT, [{"type": "text", "text": items_message}])
    except Exception as e:
        logger.error(f"piece generation failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail=f"Style advisor is temporarily unavailable — {type(e).__name__}: {e}")

    try:
        item_descriptions = _parse_items_json(raw_items)
    except (json.JSONDecodeError, IndexError, ValueError):
        raise HTTPException(status_code=502, detail="Could not build recommendations — please try again")

    return await _search_and_build(item_descriptions, profile, max_price=max_price)


async def _generate_exploratory(query: str, profile: Optional[UserProfile]) -> List[ShopItem]:
    """Generate 3-5 directional picks the user wouldn't normally think of."""
    max_price = _extract_max_price(query)
    try:
        context, _ = retrieve_context(query)
    except Exception:
        context = ""

    budget_line = f"\nHARD PRICE CEILING: ${max_price}. Every pick must realistically exist under this price." if max_price else ""

    items_message = f"""Here are relevant excerpts from expert menswear sources:

{context}

---

The user wants fresh, unexpected ideas: "{query}"
{format_profile(profile)}{budget_line}
Return a JSON array of 3-5 directional picks — each one should broaden their taste without being costumey. Remember: ONLY output the JSON array."""

    try:
        raw_items = ask_claude(EXPLORATORY_ITEMS_PROMPT, [{"type": "text", "text": items_message}])
    except Exception as e:
        logger.error(f"exploratory generation failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail=f"Style advisor is temporarily unavailable — {type(e).__name__}: {e}")

    try:
        item_descriptions = _parse_items_json(raw_items)
    except (json.JSONDecodeError, IndexError, ValueError):
        raise HTTPException(status_code=502, detail="Could not build ideas — please try again")

    return await _search_and_build(item_descriptions, profile, max_price=max_price)


@app.post("/query", response_model=QueryResponse)
async def unified_query(req: QueryRequest):
    """Unified entry point: classifies intent and routes to piece/outfit/exploratory."""
    mode = classify_intent(req.query)

    if mode == "clarify":
        return QueryResponse(
            mode="clarify",
            query=req.query,
            clarify="Can you tell me more? Are you shopping for a specific item, a full outfit for an occasion, or just looking for fresh ideas?",
        )

    if mode == "piece":
        items = await _generate_piece(req.query, req.profile)
    elif mode == "exploratory":
        items = await _generate_exploratory(req.query, req.profile)
    else:  # outfit (default)
        items = await _generate_outfit(req.query, req.profile)

    return QueryResponse(mode=mode, query=req.query, items=items)


@app.post("/shop-vibe", response_model=VibeResponse)
async def get_vibe_recommendations(req: VibeRequest):
    """Legacy outfit endpoint — kept for backward compatibility."""
    items = await _generate_outfit(req.vibe, req.profile)
    return VibeResponse(vibe=req.vibe, items=items)


@app.post("/shop-refresh", response_model=ShopItem)
async def refresh_product(req: RefreshRequest):
    """Get an alternative product for the same item category, excluding previously seen links."""
    item_info = {"item": req.item}
    loop = asyncio.get_event_loop()
    sr = await loop.run_in_executor(
        _executor, search_product, item_info, req.budget, req.exclude_links, "", req.max_price
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


