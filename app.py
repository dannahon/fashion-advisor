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
import time
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

You MUST respond with ONLY a valid JSON array of 5-8 objects. No markdown, no explanation, no preamble — just the JSON array.

Each object must have:
- "item": a specific garment or accessory description (style, color, fabric, fit).
- "slot": one of "outerwear", "top", "bottom", "shoes", or "accessory".
- "brand": a specific brand that is authentic to this item AND this occasion. Commit to a brand that actually makes a great version of this piece within the relevant cultural/regional tradition.
- "product": if you know a specific product/model name from that brand, include it (e.g. "K-Jacket", "Jackson Boot"). Otherwise empty string.

CULTURAL AUTHENTICITY — this is the most important instruction:
Commit to pieces and brands that match the actual regional or cultural tradition of the occasion. Do NOT default to generic Italian coastal prep for every query. Match the setting:
- Texas ranch wedding → western wear: pearl-snap shirts, dark selvedge denim or dressy western trousers, western boots, tooled-leather belt with western buckle, bolo tie, felt cowboy hat. Brands like Freenote Cloth, Tecovas, Lucchese, Tellason, Stetson, Yellowstone Ranch Store, Nocona.
- Italian Riviera wedding → Milanese/coastal tailoring: unstructured linen blazer, open-collar linen shirt, pleated linen trousers, tassel loafers. Brands like Boglioli, Luca Faloni, Incotex, Drake's, Crockett & Jones.
- Japanese archival streetwear → Needles, Visvim, Junya Watanabe, Comme des Garçons, WTAPS.
- PNW tech worker → Patagonia, Filson, Danner, Outdoor Research, Arc'teryx.
- Los Angeles dinner → Buck Mason, Aime Leon Dore, Bode, John Elliott.
Pull from your knowledge of the menswear tradition for wherever and whatever the user is describing — the examples above are a pattern to follow, not a closed list.

SLOT RULES:
- "outerwear": jackets, blazers, coats, vests, overshirts. Skip if not needed for the vibe.
- "top": shirts, t-shirts, polos, sweaters, henleys — anything worn on the upper body as the main layer.
- "bottom": trousers, pants, jeans, chinos, shorts.
- "shoes": any footwear — sneakers, loafers, boots, dress shoes.
- "accessory": small add-ons like belts, ties, bolo ties, pocket squares, watches, sunglasses, hats, cowboy hats, scarves, socks. A polo, sweater, or knit shirt is NEVER an accessory — those are "top".

CRITICAL — these three slots are MANDATORY in every output, no exceptions:
- Exactly 1 object with "slot": "top"
- Exactly 1 object with "slot": "bottom"
- Exactly 1 object with "slot": "shoes"
Never omit any of these, even if you think the wearer "obviously" already owns one (a basic white tee, dark jeans, etc). The user sees ONLY what you output — a missing slot means a missing piece in their outfit. If a slot would be a generic basic for this vibe, include it anyway with a brand commitment to the best version of that basic.

Cover a full outfit: outerwear (if needed), exactly 1 top, exactly 1 bottom, exactly 1 pair of shoes, and 1-3 accessories. Use 3 accessories when the occasion authentically calls for it (e.g. belt + bolo + cowboy hat for a ranch wedding, belt + watch + pocket square for black-tie). Do NOT force accessories that don't belong.

Scenario specifics:
- Boat day / beach / pool → swim trunks as the bottom, skip belts, skip outerwear
- Gym / workout → athletic shorts or joggers, performance top, trainers
- Black tie → tuxedo pieces, dress shoes, bow tie
- Casual → don't force a belt or tie if they don't add to the look

Example — Texas ranch wedding:
[{"item": "pearl-snap chambray western shirt", "slot": "top", "brand": "Freenote Cloth", "product": "Calico Western Shirt"}, {"item": "dark indigo selvedge denim slim straight", "slot": "bottom", "brand": "Tellason", "product": "Stock Selvedge"}, {"item": "tan cowhide western roper boots", "slot": "shoes", "brand": "Tecovas", "product": "The Jackson"}, {"item": "tooled tan leather belt with western buckle", "slot": "accessory", "brand": "Lucchese", "product": ""}, {"item": "silver-tipped braided leather bolo tie", "slot": "accessory", "brand": "Yellowstone Ranch Store", "product": ""}, {"item": "sand felt open-road cowboy hat", "slot": "accessory", "brand": "Stetson", "product": "Open Road"}]

Example — Italian Riviera wedding:
[{"item": "unstructured navy linen K-jacket", "slot": "outerwear", "brand": "Boglioli", "product": "K-Jacket"}, {"item": "white linen spread-collar shirt", "slot": "top", "brand": "Luca Faloni", "product": "Versilia Linen Shirt"}, {"item": "cream cotton-linen pleated trousers", "slot": "bottom", "brand": "Incotex", "product": ""}, {"item": "brown suede tassel loafers", "slot": "shoes", "brand": "Crockett & Jones", "product": "Cavendish"}, {"item": "navy silk knit tie", "slot": "accessory", "brand": "Drake's", "product": ""}]
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
    shoeSize: Optional[str] = None
    likedBrands: Optional[List[str]] = None
    dislikedBrands: Optional[List[str]] = None


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

# Country-code TLDs we reject because they price in a non-USD currency.
# Checked as hostname suffix via .endswith() so we don't false-positive on
# things like .jpg in a URL path.
_FOREIGN_TLD_ENDS = (
    '.kr', '.jp', '.cn', '.tw', '.hk', '.sg', '.in', '.au', '.nz',
    '.uk', '.ie', '.de', '.fr', '.it', '.es', '.pt', '.nl', '.be',
    '.ch', '.at', '.dk', '.se', '.no', '.fi', '.pl', '.ru', '.ua',
    '.br', '.mx', '.ar', '.cl', '.pe', '.tr', '.il', '.za', '.ae',
    '.sa', '.th', '.vn', '.id', '.ph', '.my', '.pk',
)


def _is_non_us_hostname(url: str) -> bool:
    """True if the hostname ends in a non-US country TLD."""
    try:
        host = (_urlparse(url).hostname or '').lower()
    except Exception:
        return False
    return host.endswith(_FOREIGN_TLD_ENDS)

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

# Last-resort generic queries by slot. Used only when every other search
# tier has failed for an essential slot — these are the dead-simple "you
# will not be shirtless" queries that any major US retailer stocks.
_GENERIC_SLOT_QUERIES = {
    "top": "men's button-down shirt",
    "bottom": "men's chinos",
    "shoes": "men's leather dress shoes",
    "outerwear": "men's jacket",
}


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


def _is_category_url(url: str) -> bool:
    """Check if a URL looks like a category/collection page rather than a product page."""
    lower = url.lower()
    # If the URL clearly points to a product, skip category checks entirely.
    # (handles cases like mrporter.com/en-us/mens/product/... where /mens/ would
    # otherwise false-positive as a category signal)
    product_signals = ("/product/", "/products/", "/prod/", "/item/", "/itm/")
    if any(s in lower for s in product_signals):
        return False
    category_signals = (
        "/collections/", "/collection/", "/category/", "/categories/",
        "/shop/", "/shop-by/", "/search", "/browse/", "/c/", "/plp/",
        "/buy/", "/mens-", "/womens-", "/all-",
        "/mens/", "/womens/", "/accessories/", "/clothing/",
        "/shoes/", "/bags/", "/sale/",
        "/men/", "/women/", "/designer/", "/designers/", "/brand/", "/brands/",
        "?q=", "?query=", "?search=",
    )
    # Also catch URLs that end at a category level with no product slug
    # e.g. mrporter.com/en-kw/mens/accessories/hats/caps
    segments = [s for s in lower.rstrip("/").split("/") if s]
    # If last segment is a generic category word, it's probably a listing
    category_endings = {
        "caps", "hats", "shirts", "pants", "jeans", "shoes", "boots",
        "sneakers", "loafers", "blazers", "jackets", "coats", "belts",
        "ties", "shorts", "sweaters", "polos", "tees", "accessories",
        "outerwear", "knitwear", "footwear", "bags", "sale", "new",
    }
    if segments and segments[-1] in category_endings:
        return True
    # URLs ending at a category level (e.g. site.com/mens/shirts)
    if any(s in lower for s in category_signals):
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
        " for men", " for women",
    )
    return any(s in lower for s in bad_signals)


def _scrape_product_image(url: str, timeout: float = 5.0) -> str:
    """Try to extract the product image from the actual product page.
    Looks for og:image, twitter:image, or common product image meta tags.
    Returns "" if the product is out of stock (detected via schema.org JSON-LD)."""
    try:
        resp = _requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        if resp.status_code != 200:
            return ""

        # Drop out-of-stock products. Most retailers (especially Shopify stores
        # like Deus, Drakes, etc.) emit schema.org JSON-LD with offers.availability.
        # Heuristic: bail if any offer says OutOfStock AND no offer says InStock
        # (this correctly handles variant products where some sizes are OOS but
        # others are still available — those pages contain both markers.)
        # IMPORTANT: scan the FULL HTML, not just the first 100KB — availability
        # markers are often in a JSON-LD block deep in the page.
        full_lower = resp.text.lower()
        if ("schema.org/outofstock" in full_lower
                and "schema.org/instock" not in full_lower):
            return ""

        html = resp.text[:100_000]  # og:image regex only needs the head

        # Try og:image first (most reliable for product pages).
        # Patterns use [^>]* rather than \s+ between <meta and attributes so they
        # match tags with extra attributes (e.g. React Helmet: <meta data-react-
        # helmet="true" property="og:image" content="...">).
        for pattern in [
            r'<meta[^>]*property=["\']og:image["\'][^>]*content=["\'](https?://[^"\']+)',
            r'<meta[^>]*content=["\'](https?://[^"\']+)["\'][^>]*property=["\']og:image',
            r'<meta[^>]*name=["\']twitter:image["\'][^>]*content=["\'](https?://[^"\']+)',
            r'<meta[^>]*content=["\'](https?://[^"\']+)["\'][^>]*name=["\']twitter:image',
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


# In-process TTL cache for search_product. Keyed on the resolved query
# inputs (item, brand, product, slot, budget, shoe_size). Skipped entirely
# when exclude_links is non-empty (the /shop-refresh path wants a *different*
# product than what's cached). Only successful results are cached.
# Evaporates on every Render deploy — that's fine for now; persistent caching
# is parked in the roadmap until traffic justifies the infra.
_SEARCH_CACHE: dict = {}
_SEARCH_CACHE_TTL = 24 * 3600  # 24 hours
_SEARCH_CACHE_MAX = 2000


def _search_cache_key(item_info, budget, shoe_size):
    if isinstance(item_info, str):
        return (item_info, "", "", "", budget, shoe_size)
    return (
        item_info.get("item", ""),
        item_info.get("brand", ""),
        item_info.get("product", ""),
        item_info.get("slot", ""),
        budget,
        shoe_size,
    )


def search_product(item_info, budget="", exclude_links=None, shoe_size="", site_filter="") -> dict:
    """Search for a real, in-stock product using SerpAPI Google organic + og:image.

    Uses Claude's committed brand/product when available for cultural specificity,
    and trusts Google ranking + a skip_domains list for curation rather than a
    hard site: filter (which previously capped Claude to a fixed retailer pool
    and killed brand-authentic picks like Tecovas, Stetson, Freenote, etc.).

    site_filter: optional 'site:foo.com OR site:bar.com' fragment that forces
    results into a curated retailer pool. Used as a last-resort retry for
    essential slots (top/bottom/shoes) that fail the open-web search. Cache
    is bypassed when this is set."""
    # Cache check — only when the caller isn't trying to *avoid* a specific
    # link AND isn't using a site filter (which would key-collide with the
    # normal open-web search but return different results).
    use_cache = not exclude_links and not site_filter
    cache_key = None
    if use_cache:
        cache_key = _search_cache_key(item_info, budget, shoe_size)
        entry = _SEARCH_CACHE.get(cache_key)
        if entry and (time.time() - entry[0]) < _SEARCH_CACHE_TTL:
            return entry[1]

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

    # Build query — prefer Claude's brand+product commitment for cultural fit
    if brand and product:
        query = f"{brand} {product}"
    elif brand:
        query = f"{brand} {item}"
    else:
        query = f"men's {item}"
    # Append shoe size for footwear queries to favor in-stock results
    if shoe_size and slot == "shoes":
        query += f" size {shoe_size}"

    # Quality filter: skip resale marketplaces, social sites, and content farms.
    # Google's ranking naturally surfaces authoritative retailers at the top.
    skip_domains = ("poshmark.com", "ebay.", "etsy.com", "pinterest.",
                    "reddit.com", "youtube.com", "facebook.com", "instagram.com",
                    "twitter.com", "x.com/", "wikipedia.org", "tiktok.com",
                    "therealreal.com", "grailed.com", "vestiairecollective.",
                    "amazon.com/dp", "walmart.com", "target.com")
    skip_paths = ("/blog/", "/blogs/", "/article/", "/wiki/", "/news/",
                  "/review/", "/magazine/", "/editorial/", "/guide/",
                  # Non-US locale path segments — these usually mean a
                  # different currency on otherwise-US-looking domains
                  # (e.g. drakes.com/en-kr/...). /en-us/ stays allowed.
                  "/en-gb/", "/en-au/", "/en-ca/", "/en-nz/", "/en-ie/",
                  "/en-sg/", "/en-hk/", "/en-in/", "/en-ae/", "/en-kr/",
                  "/en-jp/", "/en-eu/", "/en-de/", "/en-fr/", "/en-it/",
                  "/en-es/", "/en-nl/", "/en-be/", "/en-ch/", "/en-at/",
                  "/en-pl/", "/en-dk/", "/en-se/", "/en-no/", "/en-fi/",
                  "/en-br/", "/en-mx/",
                  "/fr-fr/", "/fr-ca/", "/fr-be/", "/fr-ch/",
                  "/de-de/", "/de-at/", "/de-ch/",
                  "/it-it/", "/es-es/", "/es-mx/", "/pt-br/", "/pt-pt/",
                  "/nl-nl/", "/nl-be/", "/pl-pl/", "/ru-ru/",
                  "/ko-kr/", "/ja-jp/", "/zh-cn/", "/zh-tw/", "/zh-hk/")

    result = {"query": query, "title": "", "link": "", "price": "", "image_url": ""}

    # Organic search for product link + price.
    # When site_filter is set we trade the brand-authentic open-web query for
    # a curated retailer pool — recall over precision.
    serp_q = f"{query} ({site_filter})" if site_filter else f"{query} buy"
    try:
        # gl=us + hl=en is enough to anchor results to the US Google index.
        # We deliberately do NOT pass a `location` param — SerpAPI's location
        # field expects an exact entry from their locations endpoint and a
        # bare "United States" was returning thinner / inconsistent result
        # sets, which caused the missing-essential-slot bug.
        search = GoogleSearch({
            "engine": "google",
            "q": serp_q,
            "api_key": SERPAPI_KEY,
            "num": 15,
            "gl": "us",
            "hl": "en",
        })
        data = search.get_dict()
        for r in data.get("organic_results", []):
            link = r.get("link", "")
            title = r.get("title", "")
            lower = link.lower()
            title_lower = title.lower()
            if link in exclude_set:
                continue
            if any(d in lower for d in skip_domains):
                continue
            if any(p in lower for p in skip_paths):
                continue
            if _is_non_us_hostname(link):
                continue
            if lower.rstrip("/").count("/") <= 2:
                continue
            if _is_category_url(lower):
                continue
            if _is_bad_title(title):
                continue
            # Filter out women's results (title or URL path signals)
            if ("/women/" in lower or "/womens/" in lower or "/ladies/" in lower
                    or "women" in title_lower or "ladies" in title_lower):
                # But allow if explicitly "men's" / "for men" too (hybrid listings)
                if "men's" not in title_lower and "mens" not in title_lower:
                    continue

            # Extract price.
            # By the time we're here the URL has passed the non-US hostname
            # filter and the locale-path filter, so the numeric
            # detected_extensions.price field is overwhelmingly USD. The
            # $20k sanity cap below catches anything that still slips
            # through. We do NOT gate on a literal $ in the extensions
            # strings because SerpAPI doesn't always populate them, and
            # doing so was dropping legit US results.
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

            # Sanity cap — anything over $20k for vibe shopping is almost
            # certainly a foreign-currency number that slipped past the
            # other filters (a Korean Won jacket reads as ~$1.7M, a Yen
            # one as ~$200k). Drop it rather than show a ridiculous price.
            try:
                numeric = float(price.replace("$", "").replace(",", ""))
                if numeric > 20000:
                    continue
            except ValueError:
                pass

            result["title"] = title
            result["link"] = link
            result["price"] = price
            break
    except Exception:
        pass

    # Image: scrape OG image from the product page (ensures image matches link).
    # If the page has no og:image or is out of stock, result["image_url"] stays
    # empty and the caller drops this result rather than showing a mismatched
    # stock photo from Google Images.
    if result["link"]:
        result["image_url"] = _scrape_product_image(result["link"])

    # Cache successful results only (link + price + image_url all present).
    # Failures are not cached so transient stock issues don't persist for 24h.
    if use_cache and cache_key and result.get("link") and result.get("price") and result.get("image_url"):
        if len(_SEARCH_CACHE) >= _SEARCH_CACHE_MAX:
            # Evict oldest entry — simple LRU-by-insertion-time
            oldest_key = min(_SEARCH_CACHE, key=lambda k: _SEARCH_CACHE[k][0])
            _SEARCH_CACHE.pop(oldest_key, None)
        _SEARCH_CACHE[cache_key] = (time.time(), result)

    # Per-call diagnostic so we can tell which search tier failed for which
    # slot. Logs the actual SerpAPI query, the slot, and which of the three
    # required fields (link/price/image) were filled. Reading Render logs
    # for `search_product:` is the quickest way to debug a missing slot.
    logger.info(
        "search_product: slot=%s site_filter=%s q=%r -> link=%s price=%s image=%s",
        slot or "?",
        bool(site_filter),
        serp_q,
        bool(result.get("link")),
        bool(result.get("price")),
        bool(result.get("image_url")),
    )

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

    # Brand preference hints learned from the user's save/skip history
    pref_lines = ""
    if req.profile:
        liked = req.profile.likedBrands or []
        disliked = req.profile.dislikedBrands or []
        if liked:
            pref_lines += f"\nThe user has shown they like these brands (from past saves): {', '.join(liked)}. Prefer them when they authentically fit this occasion. Do NOT force them onto vibes where they don't belong — cultural authenticity comes first."
        if disliked:
            pref_lines += f"\nThe user has shown they don't like these brands (from past skips): {', '.join(disliked)}. Avoid them unless no culturally appropriate alternative exists."

    items_message = f"""Here are relevant excerpts from menswear writing — use them to ground yourself in the cultural and stylistic context of this occasion:

{context}

---

The user wants to dress for this vibe/occasion: "{req.vibe}"
{format_profile(req.profile)}{pref_lines}
Commit to an authentic look for this specific occasion, drawing on the excerpts above AND your knowledge of regional/cultural menswear. Do NOT default to generic Italian coastal prep — match the actual tradition of the place, culture, and setting described.

Return a JSON array of 5-8 item descriptions WITH brand commitments. Remember: ONLY output the JSON array, nothing else."""

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

    # JSON validation: if Claude omitted top/bottom/shoes, re-prompt with an
    # explicit "you forgot these" message. Claude omissions are the leading
    # cause of shirtless/pantsless outfits (e.g. "first day of law school"
    # coming back with belt + pants + shoes but no shirt). The re-prompt is
    # cheap because we're only asking for the missing pieces, not the whole
    # outfit, and it produces vibe-appropriate items instead of a hardcoded
    # fallback.
    requested_slots_set = {
        i.get("slot") for i in item_descriptions if isinstance(i, dict)
    }
    missing_essentials = [
        s for s in ("top", "bottom", "shoes") if s not in requested_slots_set
    ]
    if missing_essentials:
        logger.warning(
            f"shop-vibe '{req.vibe}': Claude omitted {missing_essentials} from JSON — re-prompting"
        )
        fixup_message = f"""You generated an outfit for "{req.vibe}" but you forgot to include these required slots: {', '.join(missing_essentials)}.

Return ONLY a JSON array containing exactly {len(missing_essentials)} new object{'s' if len(missing_essentials) > 1 else ''} — one for each missing slot listed above. Use the same format as before (item, slot, brand, product) and match the vibe and cultural tradition of the occasion. Do NOT repeat any of the pieces from your previous output.

Here is what you returned previously (for context, do not repeat these):
{json.dumps(item_descriptions, indent=2)}

Output ONLY the JSON array of {len(missing_essentials)} new items, nothing else."""
        try:
            raw_fixup = ask_claude(VIBE_ITEMS_PROMPT, [{"type": "text", "text": fixup_message}])
            cleaned_fixup = raw_fixup.strip()
            if cleaned_fixup.startswith("```"):
                cleaned_fixup = cleaned_fixup.split("\n", 1)[1]
                cleaned_fixup = cleaned_fixup.rsplit("```", 1)[0]
            fstart = cleaned_fixup.find("[")
            fend = cleaned_fixup.rfind("]")
            if fstart != -1 and fend != -1:
                cleaned_fixup = cleaned_fixup[fstart:fend + 1]
            fixup_items = json.loads(cleaned_fixup)
            # Merge fixup items into the main list, keeping only ones that
            # actually fill a missing slot (Claude sometimes over-generates).
            for fi in fixup_items:
                if isinstance(fi, dict) and fi.get("slot") in missing_essentials:
                    item_descriptions.append(fi)
                    missing_essentials.remove(fi.get("slot"))
            logger.info(
                f"shop-vibe '{req.vibe}': re-prompt filled {len(fixup_items)} slots, still missing={missing_essentials}"
            )
        except Exception as e:
            logger.error(f"shop-vibe re-prompt failed: {type(e).__name__}: {e}")

    # Phase 2: Search for real, in-stock products for each item
    budget = req.profile.budget if req.profile else ""
    shoe_size = req.profile.shoeSize if req.profile else ""
    loop = asyncio.get_event_loop()
    search_tasks = [
        loop.run_in_executor(_executor, search_product, item_info, budget, None, shoe_size)
        for item_info in item_descriptions
    ]
    search_results = await asyncio.gather(*search_tasks)

    # Build ShopItems from search results
    items = []
    failed_essential = []  # track essential slots that failed
    for item_info, sr in zip(item_descriptions, search_results):
        if isinstance(item_info, str):
            s_item = item_info
            slot = "accessory"
            item_brand = ""
        else:
            s_item = item_info.get("item", "")
            slot = item_info.get("slot", "accessory")
            item_brand = item_info.get("brand", "")

        if not sr["link"] or not sr.get("price") or not sr.get("image_url"):
            # If an essential slot failed, queue it for retry
            if slot in ("top", "bottom", "shoes"):
                failed_essential.append(item_info)
            continue

        items.append(ShopItem(
            name=sr["title"],
            brand=item_brand,
            price=sr["price"],
            link=sr["link"],
            description=s_item,
            image=sr["image_url"],
            search_item=s_item,
            search_product="",
            slot=slot,
        ))

    # Retry failed essential slots once — strip brand commitment so the retry
    # falls back to a generic "men's <item>" search. Claude's first-pick brand
    # may not be online or may be out of stock, and retrying with the same
    # brand/product terms would just return the same empty result.
    if failed_essential:
        retry_items = [
            {**i, "brand": "", "product": ""} if isinstance(i, dict) else i
            for i in failed_essential
        ]
        retry_tasks = [
            loop.run_in_executor(_executor, search_product, item_info, budget, None, shoe_size)
            for item_info in retry_items
        ]
        retry_results = await asyncio.gather(*retry_tasks)
        still_failed = []
        for item_info, sr in zip(failed_essential, retry_results):
            if not sr["link"] or not sr.get("price") or not sr.get("image_url"):
                still_failed.append(item_info)
                continue
            s_item = item_info if isinstance(item_info, str) else item_info.get("item", "")
            slot = "accessory" if isinstance(item_info, str) else item_info.get("slot", "accessory")
            # Brand is empty here because the retry stripped Claude's commitment;
            # the actual product brand isn't easily extractable from the search
            # result title, so we leave it blank rather than guess.
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

        # Third retry — fallback for essentials still missing.
        # Forces results into the curated RETAILERS pool via a site: filter.
        # Loses cultural specificity but has very high recall for basics
        # like a white tee or navy chinos, which is exactly what we need
        # when the open-web search has already failed twice.
        truly_failed = []
        if still_failed:
            fallback_items = [
                {**i, "brand": "", "product": ""} if isinstance(i, dict) else i
                for i in still_failed
            ]
            fallback_tasks = []
            for fi in fallback_items:
                hint = fi.get("item", "") if isinstance(fi, dict) else fi
                sf = _get_site_query(budget=budget, n=8, item_hint=hint)
                fallback_tasks.append(
                    loop.run_in_executor(
                        _executor, search_product, fi, budget, None, shoe_size, sf
                    )
                )
            fallback_results = await asyncio.gather(*fallback_tasks)
            for item_info, sr in zip(still_failed, fallback_results):
                if not sr["link"] or not sr.get("price") or not sr.get("image_url"):
                    truly_failed.append(item_info)
                    continue
                s_item = item_info if isinstance(item_info, str) else item_info.get("item", "")
                slot = "accessory" if isinstance(item_info, str) else item_info.get("slot", "accessory")
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

        # Fourth retry — absolute last resort. The user must never be told
        # to attend law school in just pants and shoes. Drop Claude's whole
        # item description and search for a dead-simple slot-generic query
        # like "men's button-down shirt" that any major US retailer stocks.
        if truly_failed:
            generic_items = []
            for fi in truly_failed:
                slot_name = "" if isinstance(fi, str) else fi.get("slot", "")
                generic_q = _GENERIC_SLOT_QUERIES.get(slot_name)
                if not generic_q:
                    # No generic query for this slot (e.g. accessory) — skip
                    generic_items.append(None)
                    continue
                generic_items.append({
                    "item": generic_q,
                    "brand": "",
                    "product": "",
                    "slot": slot_name,
                })
            generic_tasks = [
                (loop.run_in_executor(
                    _executor, search_product, gi, budget, None, shoe_size
                ) if gi is not None else None)
                for gi in generic_items
            ]
            for orig, task in zip(truly_failed, generic_tasks):
                if task is None:
                    logger.warning(
                        f"shop-vibe '{req.vibe}': all 4 search attempts failed for "
                        f"essential slot {orig if isinstance(orig, str) else orig.get('slot', '?')} "
                        f"(no generic query available)"
                    )
                    continue
                sr = await task
                slot_name = "accessory" if isinstance(orig, str) else orig.get("slot", "accessory")
                if not sr["link"] or not sr.get("price") or not sr.get("image_url"):
                    logger.warning(
                        f"shop-vibe '{req.vibe}': all 4 search attempts failed for "
                        f"essential slot {slot_name}"
                    )
                    continue
                s_item = orig if isinstance(orig, str) else orig.get("item", "")
                items.append(ShopItem(
                    name=sr["title"],
                    brand="",
                    price=sr["price"],
                    link=sr["link"],
                    description=s_item,
                    image=sr["image_url"],
                    search_item=s_item,
                    search_product="",
                    slot=slot_name,
                ))

    # Diagnostic: log which slots Claude asked for vs. which ones survived
    # search + retry. Tells us whether the missing-slot bug is Claude
    # omitting a slot in the JSON or search_product failing twice.
    requested_slots = [
        (i.get("slot", "?"), i.get("item", "")) if isinstance(i, dict) else ("?", i)
        for i in item_descriptions
    ]
    delivered_slots = [it.slot for it in items]
    logger.info(
        f"shop-vibe '{req.vibe}' | requested={requested_slots} | "
        f"delivered={delivered_slots}"
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


