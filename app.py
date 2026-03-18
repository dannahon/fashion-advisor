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
from concurrent.futures import ThreadPoolExecutor
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from typing import Optional, List
from pydantic import BaseModel
from PIL import Image
from serpapi import GoogleSearch
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

_executor = ThreadPoolExecutor(max_workers=8)

DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "eaf631bccd0471c91ad21097a4c78eb76978dabbc4b88a76825b4d2564c00b6f")
TOP_K = 12  # number of chunks to retrieve
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

You MUST respond with ONLY a valid JSON array of 6-8 objects. No markdown, no explanation, no preamble — just the JSON array.

Each object must have:
- "item": a specific garment or accessory description (style, color, fabric, fit)
- "brand": a specific brand that makes a great version of this item
- "product": if you know a specific product name from that brand, include it (e.g. "Ludlow Slim-Fit Suit Jacket"). Otherwise leave empty string.

These will be used to search for real, currently available products, so be specific. Use brands with strong online retail presence — J.Crew, Todd Snyder, Suitsupply, Sid Mashburn, Drake's, Percival, Spier & Mackay, Uniqlo, Bonobos, etc.

Cover a full outfit: top, bottom, shoes, and 1-2 accessories. Mix price points.

Example output:
[{"item": "unstructured navy linen blazer", "brand": "Boglioli", "product": "K-Jacket"}, {"item": "white linen spread-collar shirt", "brand": "Proper Cloth", "product": ""}, {"item": "cream cotton chinos slim fit", "brand": "Bonobos", "product": "Stretch Washed Chino"}]
"""



def get_collection():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=DB_DIR)
    # Try style_advice first, fall back to dieworkwear for backwards compat
    try:
        return client.get_collection("style_advice", embedding_function=ef)
    except Exception:
        return client.get_collection("dieworkwear", embedding_function=ef)


def retrieve_context(query: str, n_results: int = TOP_K) -> tuple:
    """Retrieve relevant chunks from the vector store. Returns (context_text, metadata_list)."""
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)

    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        title = meta.get("title", "Untitled")
        date = meta.get("date", "")
        context_parts.append(f'--- From "{title}" ({date}) ---\n{doc}')

    return "\n\n".join(context_parts), results["metadatas"][0]


@app.get("/health")
async def health_check():
    """Debug endpoint to check if all services are working."""
    status = {"server": "ok", "chromadb": "unknown", "anthropic": "unknown", "collection": "unknown"}
    # Check ChromaDB
    try:
        client = chromadb.PersistentClient(path=DB_DIR)
        collections = client.list_collections()
        status["chromadb"] = "ok"
        status["collections"] = [c.name for c in collections]
        # Check if our collection exists
        for c in collections:
            if c.name in ("style_advice", "dieworkwear"):
                col = client.get_collection(c.name)
                status["collection"] = f"{c.name}: {col.count()} chunks"
                break
    except Exception as e:
        status["chromadb"] = f"error: {str(e)}"
    # Check Anthropic key
    try:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        status["anthropic"] = f"key set ({len(key)} chars)" if key else "NO KEY SET"
    except Exception as e:
        status["anthropic"] = f"error: {str(e)}"
    # Check DB_DIR
    status["db_dir"] = DB_DIR
    status["db_dir_exists"] = os.path.exists(DB_DIR)
    if os.path.exists(DB_DIR):
        files = os.listdir(DB_DIR)
        status["db_files"] = files[:10]
    return status


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
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    return response.content[0].text


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


class ShopItem(BaseModel):
    name: str
    brand: str
    price: str
    link: str
    description: str
    image: str = ""


class VibeResponse(BaseModel):
    vibe: str
    items: List[ShopItem]


def search_product(item_info, budget="") -> dict:
    """Search for a real, in-stock product using SerpAPI Google organic + images."""
    if isinstance(item_info, str):
        item, brand, product = item_info, "", ""
    else:
        item = item_info.get("item", "")
        brand = item_info.get("brand", "")
        product = item_info.get("product", "")

    if product and brand:
        query = f"{brand} {product} men's"
    elif brand:
        query = f"{brand} {item} men's"
    else:
        query = f"men's {item}"

    # Add price hint based on budget (gentle nudge, not hard filter)
    if budget == "budget":
        query += " affordable"

    skip_domains = ("poshmark.com", "ebay.", "etsy.com", "pinterest.", "reddit.com",
                    "youtube.com", "facebook.com", "instagram.com", "twitter.com",
                    "wikipedia.org", "tiktok.com", "therealreal.com", "grailed.com")
    skip_paths = ("/blog/", "/blogs/", "/article/", "/wiki/", "/news/",
                  "/review/", "/magazine/", "/editorial/", "/guide/")

    result = {"query": query, "title": "", "link": "", "price": "", "image_url": ""}

    # Organic search for product link + price
    try:
        search = GoogleSearch({
            "engine": "google",
            "q": f"{query} buy",
            "api_key": SERPAPI_KEY,
            "num": 8,
        })
        data = search.get_dict()
        for r in data.get("organic_results", []):
            link = r.get("link", "")
            lower = link.lower()
            if any(d in lower for d in skip_domains):
                continue
            if any(p in lower for p in skip_paths):
                continue
            if lower.rstrip("/").count("/") <= 2:
                continue
            result["title"] = r.get("title", "")
            result["link"] = link
            # Extract price from rich_snippet
            rich = r.get("rich_snippet") or {}
            bottom = rich.get("bottom") or {}
            exts = bottom.get("detected_extensions") or {}
            currency = exts.get("currency", "$")
            if exts.get("price"):
                result["price"] = f"{currency}{exts['price']:g}"
            elif exts.get("price_from"):
                result["price"] = f"{currency}{exts['price_from']:g}"
            else:
                # Fallback: parse first $XX from extensions list
                for ext_str in (bottom.get("extensions") or []):
                    if "$" in ext_str:
                        result["price"] = ext_str.split(" to ")[0].split(",")[0].strip()
                        break
            break
    except Exception:
        pass

    # Image search for product photo
    try:
        img_search = GoogleSearch({
            "engine": "google_images",
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": 1,
        })
        img_data = img_search.get_dict()
        imgs = img_data.get("images_results", [])
        if imgs:
            result["image_url"] = imgs[0].get("thumbnail", "") or imgs[0].get("original", "")
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
    context, metadatas = retrieve_context(search_query)

    items_message = f"""Here are relevant excerpts from the Die, Workwear! blog:

{context}

---

The user wants to dress for this vibe/occasion: "{req.vibe}"
{format_profile(req.profile)}
Return a JSON array of 6-8 item descriptions for a complete outfit that nails this vibe. Remember: ONLY output the JSON array, nothing else."""

    raw_items = ask_claude(VIBE_ITEMS_PROMPT, [{"type": "text", "text": items_message}])

    try:
        cleaned = raw_items.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        item_descriptions = json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        raise HTTPException(status_code=502, detail="Failed to parse outfit items")

    # Phase 2: Search for real, in-stock products for each item
    budget = req.profile.budget if req.profile else ""
    loop = asyncio.get_event_loop()
    search_tasks = [
        loop.run_in_executor(_executor, search_product, item_info, budget)
        for item_info in item_descriptions
    ]
    search_results = await asyncio.gather(*search_tasks)

    # Build ShopItems directly from search results (no Phase 3 needed — SerpAPI results are structured)
    items = []
    for item_info, sr in zip(item_descriptions, search_results):
        if not sr["link"]:
            continue
        if isinstance(item_info, str):
            brand = ""
            description = item_info
        else:
            brand = item_info.get("brand", "")
            description = item_info.get("item", "")

        items.append(ShopItem(
            name=sr["title"],
            brand=brand,
            price=sr.get("price", "") or "See site",
            link=sr["link"],
            description=description,
            image=sr["image_url"],
        ))

    return VibeResponse(vibe=req.vibe, items=items)


@app.get("/health")
async def health():
    """Health check — also verifies the vector store is loaded."""
    try:
        collection = get_collection()
        count = collection.count()
        return {"status": "ok", "chunks_indexed": count, "images_indexed": len(IMAGE_INDEX)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
