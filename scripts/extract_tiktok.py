"""
Extract structured product / category data from TikTok screenshots.

Walks a folder of screenshots, sends each one to Claude Vision, classifies it
as a product card / price ladder / hero shot / other, and pulls the relevant
fields. Aggregates everything into a single JSON catalog you can ingest into
the v2 recommendation engine.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python3 scripts/extract_tiktok.py \
        "/Users/lobster/Desktop/Tiktok Posts for Buttercream" \
        data/tiktok_catalog.json

Run with --dry-run to see how many images would be processed without making
API calls.
"""

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-4-20250514"
MIN_FILE_BYTES = 10_000  # skip likely-empty screenshots (the 4KB blank one)
MAX_WORKERS = 6           # parallel API calls

EXTRACTION_PROMPT = """You are extracting structured product data from a TikTok / menswear screenshot.

Look at the image and decide which of FOUR formats it matches, then return a JSON object with the appropriate schema.

═══════════════════════════════════════════════════
FORMAT 1: product_card
═══════════════════════════════════════════════════
A single product highlight, usually with a cream/beige background, the item name and price at the top, one or more product photos, and a short italic description below. Often from a creator like @finnlately doing a brand roundup.

Return:
{
  "type": "product_card",
  "name": "<exact item name as shown>",
  "brand": "<the brand making this item — infer from the item name, the description, or visible context>",
  "price": <numeric, in USD; convert from other currencies if needed>,
  "currency_original": "<USD, EUR, GBP, AUD, etc>",
  "description": "<the descriptive text shown under the photo, verbatim>",
  "category": "<one of: top, bottom, shoes, outerwear, bag, hat, sunglasses, watch, belt, jewelry, accessory>",
  "color": "<dominant color>",
  "fabric": "<material if mentioned>",
  "lane_guess": "<one of: elevated_everyday, european_refined, downtown_intellectual, workwear, streetwear, athletic, classic_tailoring, prep — your best guess based on aesthetic>"
}

═══════════════════════════════════════════════════
FORMAT 2: price_ladder
═══════════════════════════════════════════════════
A "where to get them" comparison post showing a category name with 3-4 brands at different price tiers ($, $$, $$$). Usually has 1-2 outfit photos at the top.

Return:
{
  "type": "price_ladder",
  "category_name": "<the centered category name, e.g. 'Knit Polo'>",
  "category": "<one of: top, bottom, shoes, outerwear, bag, hat, sunglasses, watch, belt, jewelry, accessory>",
  "options": [
    {"brand": "<name>", "price": <numeric USD>, "tier": "budget|budget_plus|mid|premium"},
    ...
  ],
  "lane_guess": "<as above>"
}

═══════════════════════════════════════════════════
FORMAT 3: hero_shot
═══════════════════════════════════════════════════
A Pinterest-style or aesthetic product photo without the cream-card template. Just one product photo with maybe a name overlay. Less info-dense.

Return:
{
  "type": "hero_shot",
  "name": "<best guess at the item or brand if visible>",
  "brand": "<if identifiable>",
  "category": "<as above>",
  "color": "<dominant color>",
  "lane_guess": "<as above>"
}

═══════════════════════════════════════════════════
FORMAT 4: other
═══════════════════════════════════════════════════
Anything that doesn't fit the above — UI screenshots, blank frames, react overlays without a real product visible, etc.

Return:
{
  "type": "other",
  "note": "<one sentence describing what this actually is>"
}

═══════════════════════════════════════════════════
RULES
═══════════════════════════════════════════════════
- Output ONLY the JSON object. No markdown fences, no preamble, no explanation.
- If a price is in a non-USD currency, convert at approximate rates (€1≈$1.10, £1≈$1.27, A$1≈$0.65).
- For brand inference on product cards: if the image shows "Twill Baker Pant" with no visible logo but the format matches Buck Mason's marketing template, return "Buck Mason" as the brand.
- For "lane_guess": be specific. Use your read of the aesthetic, not the literal item.
- If you genuinely can't tell something, use null rather than guessing wildly.
"""


def encode_image(path: Path) -> tuple[str, str]:
    """Read an image and return (base64_data, media_type)."""
    suffix = path.suffix.lower()
    media_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/png")
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def extract_one(client: anthropic.Anthropic, path: Path) -> dict:
    """Send one screenshot to Claude and parse the JSON response."""
    img_data, media_type = encode_image(path)
    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_data,
                        },
                    },
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }],
        )
        raw = msg.content[0].text.strip()
        # Strip markdown fences if Claude added them despite instructions
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        # Find the JSON object
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end + 1]
        parsed = json.loads(raw)
        parsed["source_screenshot"] = path.name
        return parsed
    except json.JSONDecodeError as e:
        return {
            "type": "error",
            "source_screenshot": path.name,
            "error": f"JSON parse failed: {e}",
            "raw_response": raw[:500] if "raw" in dir() else "",
        }
    except Exception as e:
        return {
            "type": "error",
            "source_screenshot": path.name,
            "error": f"{type(e).__name__}: {e}",
        }


def deduplicate(records: list[dict]) -> list[dict]:
    """Merge duplicate product cards (same brand + name)."""
    seen_keys = {}
    deduped = []
    for r in records:
        if r.get("type") == "product_card":
            key = (
                (r.get("brand") or "").strip().lower(),
                (r.get("name") or "").strip().lower(),
            )
            if key in seen_keys:
                # Merge source_screenshot lists
                existing = seen_keys[key]
                if isinstance(existing.get("source_screenshot"), str):
                    existing["source_screenshot"] = [existing["source_screenshot"]]
                existing["source_screenshot"].append(r["source_screenshot"])
                continue
            seen_keys[key] = r
        deduped.append(r)
    return deduped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dir", type=Path, help="Folder of screenshots")
    ap.add_argument("output_path", type=Path, help="Where to write the JSON catalog")
    ap.add_argument("--dry-run", action="store_true", help="Skip API calls, just count files")
    ap.add_argument("--limit", type=int, default=0, help="Process only the first N files (debug)")
    ap.add_argument("--retry-errors", action="store_true",
                    help="Load existing output, retry only the records with type=error, "
                         "merge back in. Uses sequential calls with throttling to stay "
                         "under the rate limit.")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS,
                    help="Concurrent API calls (default 6 for fresh run, force 1 for --retry-errors)")
    ap.add_argument("--throttle", type=float, default=0.0,
                    help="Seconds to sleep between calls (use ~6 for retry to stay under rate limit)")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        print(f"ERROR: {args.input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Retry-errors mode: load existing catalog, find error records, only re-process those
    existing_records = []
    error_filenames = set()
    if args.retry_errors:
        if not args.output_path.exists():
            print(f"ERROR: --retry-errors needs {args.output_path} to exist", file=sys.stderr)
            sys.exit(1)
        with open(args.output_path) as f:
            existing = json.load(f)
        for r in existing.get("records", []):
            if r.get("type") == "error":
                error_filenames.add(r["source_screenshot"])
            else:
                existing_records.append(r)
        print(f"Found {len(error_filenames)} error records to retry, "
              f"keeping {len(existing_records)} good records", file=sys.stderr)
        if args.workers > 1:
            print(f"Forcing --workers=1 for retry mode (was {args.workers})", file=sys.stderr)
            args.workers = 1
        if args.throttle == 0.0:
            args.throttle = 6.0  # ~10 calls/min, well under 30K tokens/min limit
            print(f"Defaulting --throttle to {args.throttle}s for retry mode", file=sys.stderr)

    # Collect candidate files
    images = sorted(
        p for p in args.input_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    )
    skipped_small = [p for p in images if p.stat().st_size < MIN_FILE_BYTES]
    images = [p for p in images if p.stat().st_size >= MIN_FILE_BYTES]
    if args.retry_errors:
        images = [p for p in images if p.name in error_filenames]
    if args.limit:
        images = images[:args.limit]

    print(f"Found {len(images)} images to process", file=sys.stderr)
    if skipped_small:
        print(f"Skipped {len(skipped_small)} files under {MIN_FILE_BYTES} bytes:", file=sys.stderr)
        for p in skipped_small:
            print(f"  - {p.name} ({p.stat().st_size} bytes)", file=sys.stderr)

    if args.dry_run:
        print("Dry run — exiting without API calls.", file=sys.stderr)
        sys.exit(0)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set in env", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    results = list(existing_records)  # carry forward good records when retrying
    started = time.time()

    if args.workers <= 1:
        # Sequential path with optional throttling — for rate-limit retries
        for i, p in enumerate(images, 1):
            r = extract_one(client, p)
            results.append(r)
            kind = r.get("type", "?")
            label = r.get("name") or r.get("category_name") or r.get("note") or r.get("error") or "?"
            label_short = label[:80] if isinstance(label, str) else str(label)[:80]
            print(f"[{i}/{len(images)}] {p.name} → {kind}: {label_short}", file=sys.stderr)
            if args.throttle and i < len(images):
                time.sleep(args.throttle)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(extract_one, client, p): p for p in images}
            for i, fut in enumerate(as_completed(futures), 1):
                p = futures[fut]
                r = fut.result()
                results.append(r)
                kind = r.get("type", "?")
                label = r.get("name") or r.get("category_name") or r.get("note") or r.get("error") or "?"
                label_short = label[:80] if isinstance(label, str) else str(label)[:80]
                print(f"[{i}/{len(images)}] {p.name} → {kind}: {label_short}", file=sys.stderr)

    elapsed = time.time() - started
    print(f"\nProcessed {len(results)} images in {elapsed:.1f}s", file=sys.stderr)

    # Dedup product cards
    before = len(results)
    results = deduplicate(results)
    if len(results) != before:
        print(f"Deduplicated {before - len(results)} duplicate product cards", file=sys.stderr)

    # Sort by type for easier reading
    type_order = ["product_card", "price_ladder", "hero_shot", "other", "error"]
    results.sort(key=lambda r: (type_order.index(r.get("type", "other")) if r.get("type") in type_order else 99,
                                 r.get("brand") or "",
                                 r.get("name") or r.get("category_name") or ""))

    # Summary by type
    counts = {}
    for r in results:
        counts[r.get("type", "?")] = counts.get(r.get("type", "?"), 0) + 1
    print(f"\nSummary: {counts}", file=sys.stderr)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump({"records": results, "summary": counts}, f, indent=2)
    print(f"\nWrote {args.output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
