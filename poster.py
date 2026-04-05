#!/usr/bin/env python3
"""poster.py — Generate MM2K and CL2K style Plex media posters.

Usage:
    python poster.py "Inception"
    python poster.py "The Dark Knight" --year 2008
    python poster.py "Breaking Bad" --type show --season 1
"""

import argparse
import base64
import io
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Constants ─────────────────────────────────────────────────────────────────

POSTER_W, POSTER_H = 1000, 1500
BORDER = 25
GRADIENT_START_Y = 1065    # gradient starts here (pixel-measured from reference)
GRADIENT_DARKEST_Y = 1360  # fully opaque black from here down
GRADIENT_TOP_FRAC = 0.71   # unused legacy, kept for reference
TMDB_BASE = "https://api.themoviedb.org/3"
IMG_BASE = "https://image.tmdb.org/t/p/original"
APPLE_STOREFRONT = "143441"  # US
APPLE_URL_SERVICE = "https://itunesartwork.bendodson.com/url.php"

# Words that trigger a line split in MM2K titles
CONNECTOR_WORDS = {"OF", "AND", "VS", "IN", "AT"}

# Season number to spelled-out word (per MM2K PSD spec)
SEASON_WORDS = {
    1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE",
    6: "SIX", 7: "SEVEN", 8: "EIGHT", 9: "NINE", 10: "TEN",
    11: "ELEVEN", 12: "TWELVE", 13: "THIRTEEN", 14: "FOURTEEN",
    15: "FIFTEEN", 16: "SIXTEEN", 17: "SEVENTEEN", 18: "EIGHTEEN",
    19: "NINETEEN", 20: "TWENTY",
}

# ── TMDB helpers ──────────────────────────────────────────────────────────────

def tmdb_get(path: str, params: dict, api_key: str) -> dict:
    params = dict(params, api_key=api_key)
    r = requests.get(f"{TMDB_BASE}{path}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def search_tmdb(query: str, year: Optional[int], media_type: str, api_key: str) -> dict:
    endpoint = "/search/tv" if media_type == "show" else "/search/movie"
    params: dict = {"query": query, "language": "en-US"}
    if year:
        params["first_air_date_year" if media_type == "show" else "year"] = year
    data = tmdb_get(endpoint, params, api_key)
    results = data.get("results", [])
    if not results:
        raise ValueError(f"No TMDB results for '{query}'")
    return results[0]


def fetch_images(media_id: int, media_type: str, api_key: str) -> dict:
    path = f"/{'tv' if media_type == 'show' else 'movie'}/{media_id}/images"
    return tmdb_get(path, {"include_image_language": "en,null"}, api_key)


def fetch_external_ids(media_id: int, media_type: str, api_key: str) -> dict:
    """Fetch IMDB, TVDB IDs for correct TPDB-style filename."""
    path = f"/{'tv' if media_type == 'show' else 'movie'}/{media_id}/external_ids"
    return tmdb_get(path, {}, api_key)


def build_filename_stem(canonical: str, year: Optional[int], media_id: int,
                        ext_ids: dict, media_type: str,
                        season: Optional[int] = None) -> str:
    """Build TPDB-compliant filename stem.

    Movies:  Title (Year) {tmdb-ID} {imdb-ttXXX}
    Shows:   Title (Year) {tmdb-ID} {tvdb-ID} {imdb-ttXXX}
    Seasons: Title (Year) {tmdb-ID} {tvdb-ID} {imdb-ttXXX} - Season N
    """
    safe = canonical.replace("/", "-").replace(":", " -")
    year_str = f" ({year})" if year else ""
    tmdb_tag = f"{{tmdb-{media_id}}}"

    imdb_id = ext_ids.get("imdb_id", "")
    tvdb_id = ext_ids.get("tvdb_id", "")

    imdb_tag = f"{{imdb-{imdb_id}}}" if imdb_id else ""
    tvdb_tag = f"{{tvdb-{tvdb_id}}}" if tvdb_id else ""

    if media_type == "show":
        tags = " ".join(filter(None, [tmdb_tag, tvdb_tag, imdb_tag]))
    else:
        tags = " ".join(filter(None, [tmdb_tag, imdb_tag]))

    stem = f"{safe}{year_str} {tags}".strip()

    if season is not None:
        stem += f" - Season {season}"

    return stem


def best_background(images: dict) -> Optional[str]:
    """Select best portrait poster image, preferring textless (null language)."""
    posters = images.get("posters", [])
    backdrops = images.get("backdrops", [])

    def score(item: dict) -> tuple:
        return (item.get("vote_average", 0), item.get("width", 0))

    # Prefer textless posters (portrait, no language tag)
    textless = [p for p in posters if p.get("iso_639_1") is None]
    if textless:
        return IMG_BASE + max(textless, key=score)["file_path"]
    if posters:
        return IMG_BASE + max(posters, key=score)["file_path"]

    # Fall back to textless backdrops (landscape, will be cropped)
    textless_bd = [b for b in backdrops if b.get("iso_639_1") is None]
    if textless_bd:
        return IMG_BASE + max(textless_bd, key=score)["file_path"]
    if backdrops:
        return IMG_BASE + max(backdrops, key=score)["file_path"]

    return None


def logo_white_score(img: Image.Image) -> float:
    """Return fraction of visible logo pixels that are bright, low-saturation white."""
    import numpy as np

    arr = np.array(img.convert("RGBA"))
    visible = arr[:, :, 3] > 10
    if visible.sum() == 0:
        return 0.0
    rgb = arr[visible][:, :3].astype(float)
    r_ch, g_ch, b_ch = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.maximum(np.maximum(r_ch, g_ch), b_ch)
    minc = np.minimum(np.minimum(r_ch, g_ch), b_ch)
    sat = np.where(maxc > 0, (maxc - minc) / maxc, 0.0)
    brightness = (r_ch + g_ch + b_ch) / 3.0
    return float(((sat < 0.15) & (brightness > 200)).mean())


def best_logo(images: dict) -> Optional[str]:
    """Select best English PNG logo for CL2K style.
    
    Prefers already-white logos (low saturation) per spec ('as white as possible').
    Falls back to vote_average if no white version found.
    """
    logos = images.get("logos", [])
    if not logos:
        return None

    en_png = [l for l in logos if l.get("iso_639_1") == "en"
              and l["file_path"].lower().endswith(".png")]
    candidates = en_png if en_png else [l for l in logos if l.get("iso_639_1") == "en"]
    if not candidates:
        candidates = logos

    # Sample each logo to find the whitest one (lowest saturation)
    def logo_whiteness(item: dict) -> float:
        try:
            import io as _io
            # Use a small thumbnail from TMDB to check colour
            thumb_url = f"https://image.tmdb.org/t/p/w300{item['file_path']}"
            r = requests.get(thumb_url, timeout=10)
            img = Image.open(_io.BytesIO(r.content)).convert("RGBA")
            return logo_white_score(img)
        except Exception:
            return 0.0

    print(f"  Checking {len(candidates)} logos for whiteness …")
    scored = [(logo_whiteness(l), l.get("vote_average", 0), l) for l in candidates]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_white_frac, _, best = scored[0]
    print(f"  Best logo whiteness: {best_white_frac:.0%} | {best['file_path']}")
    return IMG_BASE + best["file_path"]


def force_white_logo_gemini(logo: Image.Image, api_key: str) -> Image.Image:
    """Use Gemini Flash vision to intelligently extract logo as clean white on transparent bg."""
    from google import genai as google_genai
    from google.genai import types as gtypes
    import json as _json
    import numpy as np

    client = google_genai.Client(api_key=api_key)

    # Convert logo to PNG for Gemini
    logo_rgba = logo.convert("RGBA")
    buf = io.BytesIO()
    logo_rgba.save(buf, format="PNG")
    buf.seek(0)
    img_bytes = buf.read()

    # Load repo guide image to give Gemini context on sizing
    try:
        with open(CL2K_GUIDE_PATH, "rb") as gf:
            guidelines_bytes = gf.read()
        guidelines_mime = "image/jpeg"
    except Exception:
        guidelines_bytes = None

    prompt = (
        "You are helping create CL2K style Plex media posters. "
        "The first image is the CL2K specification/guidelines showing how logos should be placed on a 1000x1500px poster. "
        "The second image is the logo to be placed on the poster.\n\n"
        "The logo must be placed bottom-aligned to the guideline at y=1342 on the 1500px tall poster. "
        "The spec says logo width should be 600px but can be up to 800px for complex/wide logos. "
        "The logo must be centered horizontally within the 1000px wide poster (between the 25px borders).\n\n"
        "Return ONLY a JSON object with these fields:\n"
        "{\n"
        '  "has_transparency": true/false,\n'
        '  "bg_is_dark": true/false,\n'
        '  "bg_is_light": true/false,\n'
        '  "bg_is_coloured": true/false,\n'
        '  "bg_color_rgb": [r, g, b],\n'
        '  "text_is_light": true/false,\n'
        '  "text_color_rgb": [r, g, b],\n'
        '  "saturation_threshold": 0.0-1.0,\n'
        '  "brightness_threshold": 0-255,\n'
        '  "recommended_width_px": 600-800\n'
        "}"
    )

    contents = []
    if guidelines_bytes:
        contents.append(gtypes.Part.from_bytes(data=guidelines_bytes, mime_type="image/jpeg"))
    contents.append(gtypes.Part.from_bytes(data=img_bytes, mime_type="image/png"))
    contents.append(prompt)

    response = client.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=contents,
    )

    text = response.text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        info = _json.loads(text)
    except Exception:
        # Fallback to PIL method if parsing fails
        print(f"  Gemini response parse failed, falling back to PIL")
        return force_white_logo(logo)

    recommended_width = int(info.get("recommended_width_px", 600))
    print(f"  Gemini logo analysis: bg_coloured={info.get('bg_is_coloured')}, text_light={info.get('text_is_light')}, sat_thresh={info.get('saturation_threshold')}, bright_thresh={info.get('brightness_threshold')}, width={recommended_width}px")

    arr = np.array(logo_rgba)
    r, g, b, a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]
    visible = a > 10
    brightness = (r.astype(float) + g.astype(float) + b.astype(float)) / 3.0

    result = np.zeros_like(arr)
    result[:,:,0] = 255
    result[:,:,1] = 255
    result[:,:,2] = 255

    if info.get("has_transparency") and info.get("text_is_light"):
        # Already transparent bg with light/white text — just paint white
        result[:,:,3] = a
    elif info.get("bg_is_coloured") or (not info.get("text_is_light", True)):
        # Coloured background: use saturation to isolate low-sat foreground
        sat_thresh = float(info.get("saturation_threshold", 0.20))
        bright_thresh = float(info.get("brightness_threshold", 150))
        # Relax thresholds — Gemini tends to be conservative
        sat_thresh = max(sat_thresh, 0.15)  # at least 0.15 saturation tolerance
        bright_thresh = min(bright_thresh, 200)  # at most 200 brightness floor
        rf, gf, bf = r.astype(float), g.astype(float), b.astype(float)
        maxc = np.maximum(np.maximum(rf, gf), bf)
        minc = np.minimum(np.minimum(rf, gf), bf)
        sat = np.where(maxc > 0, (maxc - minc) / maxc, 0.0)
        letter_mask = visible & (sat < sat_thresh) & (brightness > bright_thresh)
        print(f"  Logo mask pixels: {letter_mask.sum()} (sat<{sat_thresh:.2f}, bright>{bright_thresh:.0f})")
        if letter_mask.sum() < 500:
            # Too few pixels — progressively relax until we get something usable
            for sat_t, bri_t in [(0.25, 180), (0.35, 160), (0.5, 140)]:
                letter_mask = visible & (sat < sat_t) & (brightness > bri_t)
                print(f"  Relaxed to sat<{sat_t}, bright>{bri_t}: {letter_mask.sum()} px")
                if letter_mask.sum() > 500:
                    break
        if letter_mask.sum() < 100:
            result[:,:,3] = a
        else:
            from scipy.ndimage import binary_erosion, label as ndlabel
            try:
                # Erode 2px to remove anti-aliasing fringe
                letter_mask = binary_erosion(letter_mask, iterations=2)
                # Remove small isolated blobs (artifacts) — keep only components > 200px
                labeled, num_features = ndlabel(letter_mask)
                sizes = np.bincount(labeled.ravel())
                sizes[0] = 0  # ignore background
                keep = sizes >= 200
                letter_mask = keep[labeled]
            except ImportError:
                pass
            result[:,:,3] = np.where(letter_mask, np.uint8(255), np.uint8(0))
    elif info.get("bg_is_dark"):
        # Dark background with lighter letters — use brightness threshold
        bright_thresh = float(info.get("brightness_threshold", 160))
        letter_mask = visible & (brightness > bright_thresh)
        result[:,:,3] = np.where(letter_mask, np.uint8(255), np.uint8(0))
    else:
        # Light background, dark letters — invert: paint bg transparent, letters white
        bright_thresh = float(info.get("brightness_threshold", 200))
        bg_mask = visible & (brightness > bright_thresh)
        letter_mask = visible & ~bg_mask
        if letter_mask.sum() < 100:
            result[:,:,3] = a
        else:
            result[:,:,3] = np.where(letter_mask, np.uint8(255), np.uint8(0))

    return Image.fromarray(result, "RGBA"), recommended_width




def download_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGBA")


def fetch_apple_tv_item(
    title: str,
    media_type: str,
    storefront: str = APPLE_STOREFRONT,
) -> Optional[dict]:
    """Fetch the best-matching Apple TV item metadata via Ben Dodson's artwork helper."""
    # Step 1: get Apple API URL with fresh utsk token from Ben Dodson's service
    resp = requests.post(
        APPLE_URL_SERVICE,
        data={"query": title, "storefront": storefront, "locale": "en-US"},
        timeout=10,
    )
    resp.raise_for_status()
    apple_url = resp.json().get("url")
    if not apple_url:
        return None

    # Step 2: query Apple UTS API
    resp2 = requests.get(apple_url, timeout=10)
    resp2.raise_for_status()
    data = resp2.json()

    shelves = data.get("data", {}).get("canvas", {}).get("shelves", [])
    apple_type = "Show" if media_type == "show" else "Movie"
    title_norm = title.strip().lower()

    matched_item = None
    for shelf in shelves:
        for item in shelf.get("items", []):
            if item.get("type") != apple_type:
                continue
            item_title = item.get("title", "").strip().lower()
            if item_title == title_norm:
                matched_item = item
                break
        if matched_item:
            break

    # Fallback: startswith match
    if not matched_item:
        for shelf in shelves:
            for item in shelf.get("items", []):
                if item.get("type") != apple_type:
                    continue
                item_title = item.get("title", "").strip().lower()
                if item_title.startswith(title_norm) or title_norm.startswith(item_title):
                    matched_item = item
                    break
            if matched_item:
                break

    return matched_item


def apple_image_url(image_data: dict, fmt: str = "jpg") -> Optional[str]:
    if not image_data:
        return None
    url_template = image_data.get("url", "")
    if not url_template:
        return None
    w = image_data.get("width", 1000)
    h = image_data.get("height", 1000)
    return (
        url_template
        .replace("{w}", str(w))
        .replace("{h}", str(h))
        .replace("{c}", "")
        .replace("{f}", fmt)
    )


def fetch_apple_tv_logo(
    title: str,
    media_type: str,
    storefront: str = APPLE_STOREFRONT,
) -> Optional[Image.Image]:
    """Try to fetch a SingleColorContentLogo from Apple TV artwork API.

    Returns a white-on-transparent RGBA Image, or None if not found.
    """
    try:
        item = fetch_apple_tv_item(title, media_type, storefront)
        if not item:
            return None
        images = item.get("images", {})
        logo_data = images.get("singleColorContentLogo") or images.get("SingleColorContentLogo")
        logo_url = apple_image_url(logo_data, fmt="png")
        if not logo_url:
            return None
        return download_image(logo_url)
    except Exception as e:
        print(f"  Apple TV logo lookup failed ({e}) — will try TMDB fallback.")
        return None


def fetch_apple_tv_background(
    title: str,
    media_type: str,
    storefront: str = APPLE_STOREFRONT,
) -> Optional[Image.Image]:
    """Try to fetch Apple TV key art for use as poster background.

    Prefer the wider `CenteredFullScreenBackgroundImage`, then fall back to the
    smaller portrait variant.
    """
    try:
        item = fetch_apple_tv_item(title, media_type, storefront)
        if not item:
            return None
        images = item.get("images", {})
        bg_data = (
            images.get("centeredFullScreenBackgroundImage")
            or images.get("CenteredFullScreenBackgroundImage")
            or images.get("centeredFullScreenBackgroundSmallImage")
            or images.get("CenteredFullScreenBackgroundSmallImage")
            or images.get("fullScreenBackgroundSmallImage")
            or images.get("FullScreenBackgroundSmallImage")
        )
        bg_url = apple_image_url(bg_data, fmt="jpg")
        if not bg_url:
            return None
        return download_image(bg_url)
    except Exception as e:
        print(f"  Apple TV background lookup failed ({e}) — will try TMDB fallback.")
        return None


def get_release_year(result: dict, media_type: str) -> Optional[int]:
    key = "first_air_date" if media_type == "show" else "release_date"
    date = result.get(key, "") or ""
    return int(date[:4]) if len(date) >= 4 and date[:4].isdigit() else None

# ── OpenAI inpainting ─────────────────────────────────────────────────────────

def inpaint_remove_text(img: Image.Image, api_key: str) -> Image.Image:
    """Use OpenAI gpt-image-1 to remove text/logos from top 20% and bottom 30%."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    # Resize to 1024x1536 (closest supported size to 1000x1500)
    work_size = (1024, 1536)
    work = img.convert("RGBA").resize(work_size, Image.LANCZOS)
    w, h = work.size

    # Build mask: alpha=0 (transparent) → inpaint; alpha=255 (opaque) → keep
    mask = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    draw = ImageDraw.Draw(mask)
    draw.rectangle([0, 0, w, int(h * 0.20)], fill=(0, 0, 0, 0))       # top 20%
    draw.rectangle([0, int(h * 0.70), w, h], fill=(0, 0, 0, 0))       # bottom 30%

    img_buf = io.BytesIO()
    work.save(img_buf, format="PNG")
    img_buf.seek(0)

    mask_buf = io.BytesIO()
    mask.save(mask_buf, format="PNG")
    mask_buf.seek(0)

    response = client.images.edit(
        model="gpt-image-1",
        image=("image.png", img_buf, "image/png"),
        mask=("mask.png", mask_buf, "image/png"),
        prompt=(
            "A clean, textless cinematic movie or TV show poster background. "
            "Remove all text, titles, logos, watermarks, and overlays. "
            "Pure atmospheric background art, no typography."
        ),
        n=1,
    )

    item = response.data[0]
    # gpt-image-1 returns b64_json; DALL-E 2 returns url
    if hasattr(item, "b64_json") and item.b64_json:
        img_bytes = base64.b64decode(item.b64_json)
    elif hasattr(item, "url") and item.url:
        img_bytes = requests.get(item.url, timeout=30).content
    else:
        raise RuntimeError("Unexpected inpainting response format")

    inpainted = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    # Scale back to original size
    return inpainted.resize(img.size, Image.LANCZOS)

# ── Image utilities ───────────────────────────────────────────────────────────

def fit_to_poster(img: Image.Image, focus_x: float = 0.5, focus_y: float = 0.5) -> Image.Image:
    """Scale and crop image to exactly POSTER_W × POSTER_H around a focus point.

    focus_x/focus_y are normalized 0..1 coordinates in the resized image.
    """
    tw, th = POSTER_W, POSTER_H
    iw, ih = img.size
    scale = max(tw / iw, th / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = img.resize((nw, nh), Image.LANCZOS)

    fx = max(0.0, min(1.0, float(focus_x)))
    fy = max(0.0, min(1.0, float(focus_y)))
    left = int(round(fx * nw - tw / 2))
    top = int(round(fy * nh - th / 2))
    left = max(0, min(left, nw - tw))
    top = max(0, min(top, nh - th))
    return resized.crop((left, top, left + tw, top + th))


def suggest_background_focus(
    img: Image.Image,
    title: str,
    media_type: str,
    source: str = "tmdb",
) -> tuple[float, float]:
    """Return a normalized crop focus point for poster framing.

    Simple mode: use a centered crop. For Apple art, the wider background variant
    usually gives enough breathing room without extra AI steering.
    """
    return 0.5, 0.5


def make_gradient_overlay(width: int, height: int) -> Image.Image:
    """Black gradient precisely matching MM2K reference spec.
    
    Pixel-measured from reference posters:
    - Starts transparent at y=1065
    - Reaches full black at y=1360
    - Fully opaque below y=1360
    """
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    start_y = GRADIENT_START_Y
    full_y = GRADIENT_DARKEST_Y
    span = full_y - start_y
    for y in range(start_y, height):
        if y >= full_y:
            alpha = 255
        else:
            t = (y - start_y) / span
            # Linear ramp matching reference profile
            alpha = int(255 * t)
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
    return overlay


def add_white_border(img: Image.Image) -> Image.Image:
    """Draw solid 7px white border on all sides per MM2K spec."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    # Draw a solid filled rectangle frame (more reliable than outline loop)
    # Top
    draw.rectangle([0, 0, w, BORDER - 1], fill=(255, 255, 255, 255))
    # Bottom
    draw.rectangle([0, h - BORDER, w, h], fill=(255, 255, 255, 255))
    # Left
    draw.rectangle([0, 0, BORDER - 1, h], fill=(255, 255, 255, 255))
    # Right
    draw.rectangle([w - BORDER, 0, w, h], fill=(255, 255, 255, 255))
    return img

# ── Font helpers ──────────────────────────────────────────────────────────────

_FONT_PATHS = [
    # Arial Bold — user-installed (preferred)
    os.path.expanduser("~/.local/share/fonts/Arial_Bold.ttf"),
    os.path.expanduser("~/.local/share/fonts/Arial Bold.ttf"),
    # Arial (system-wide Windows / macOS / Linux with MS fonts)
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    "/usr/share/fonts/truetype/arial.ttf",
    # Liberation Sans Bold — metric-compatible fallback
    "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    # DejaVu Bold — last resort
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    # Absolute last resort — tiny bitmap font, won't scale
    return ImageFont.load_default()


def measure_text(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def fit_font(draw: ImageDraw.Draw, text: str, max_width: int,
             start_size: int = 120, min_size: int = 24) -> ImageFont.FreeTypeFont:
    """Return largest font size where text fits within max_width."""
    size = start_size
    while size >= min_size:
        font = load_font(size)
        w, _ = measure_text(draw, text, font)
        if w <= max_width:
            return font
        size -= 4
    return load_font(min_size)


_FONT_PATHS_REGULAR = [
    os.path.expanduser("~/.local/share/fonts/Arial.ttf"),
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

CL2K_GUIDE_PATH = Path(__file__).resolve().parent / "assets" / "cl2k-guide.jpg"


def load_font_regular(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_PATHS_REGULAR:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return load_font(size)  # fall back to bold if regular not found


def draw_tracked(
    draw: ImageDraw.Draw,
    cx: int,
    text: str,
    y: int,
    font: ImageFont.FreeTypeFont,
    tracking_px: int = 26,
    alpha: int = 230,
):
    """Draw text centred at cx with manual character tracking (letter-spacing)."""
    total_w = 0
    char_widths = []
    for ch in text:
        bbox = draw.textbbox((0, 0), ch, font=font)
        cw = bbox[2] - bbox[0]
        char_widths.append(cw)
        total_w += cw
    total_w += tracking_px * (len(text) - 1)

    x = cx - total_w // 2
    for ch, cw in zip(text, char_widths):
        draw.text((x, y), ch, font=font, fill=(255, 255, 255, alpha), anchor="lm")
        x += cw + tracking_px


def extract_cl2k_guide_spec(path: Path = CL2K_GUIDE_PATH) -> dict:
    """Read the CL2K guide image and derive the usable logo guide box.

    The guide image is a 1000px-wide crop of the poster bottom area, so x coordinates
    map directly to poster space. For y, we align the guide's 'Gradient Darkest Line'
    to the known poster y=1360 and derive the other rows from that offset.
    """
    fallback = {
        "main_left": 200,
        "main_right": 800,
        "max_left": 115,
        "max_right": 885,
        "top": 1095,
        "bottom": 1342,
        "gradient_darkest": 1360,
    }
    try:
        import numpy as np
        guide = Image.open(path).convert("RGB")
        arr = np.array(guide)

        # Cyan guide-line detector (JPEG-tolerant)
        cyan = (arr[:, :, 1] > 120) & (arr[:, :, 2] > 120) & (arr[:, :, 0] < 140)
        row_counts = cyan.sum(axis=1)
        col_counts = cyan.sum(axis=0)

        def cluster_positions(counts, min_count):
            pos = [i for i, c in enumerate(counts) if c >= min_count]
            if not pos:
                return []
            groups = [[pos[0]]]
            for p in pos[1:]:
                if p <= groups[-1][-1] + 2:
                    groups[-1].append(p)
                else:
                    groups.append([p])
            return [int(round(sum(g) / len(g))) for g in groups]

        vlines = cluster_positions(col_counts, 120)
        hlines = cluster_positions(row_counts, 300)

        # Expected verticals from the guide image:
        # max-left ~116, main-left ~212, centre ~501, main-right ~789, max-right ~886
        left_candidates = sorted([x for x in vlines if x < 450])
        right_candidates = sorted([x for x in vlines if x > 550])
        max_left = left_candidates[0] if left_candidates else fallback["max_left"]
        main_left = left_candidates[1] if len(left_candidates) > 1 else fallback["main_left"]
        main_right = right_candidates[0] if right_candidates else fallback["main_right"]
        max_right = right_candidates[1] if len(right_candidates) > 1 else fallback["max_right"]

        # Expected horizontals from the guide image:
        # top ~95, main-logo-bottom ~338, gradient-darkest ~360 (white line)
        top_guide = min(hlines) if hlines else 95
        main_bottom_guide = max(hlines) if hlines else 338

        white = (arr[:, :, 0] > 150) & (arr[:, :, 1] > 150) & (arr[:, :, 2] > 150)
        white_rows = white[:, 50:950].sum(axis=1)
        gradient_darkest_guide = int(np.argmax(white_rows[250:420]) + 250)

        # The guide labels the gradient-darkest line; keep that anchored at poster y=1360.
        # For the main-logo bottom baseline, preserve the reference-calibrated poster y=1342.
        # The guide image then supplies the logo-box height above that baseline.
        guide_logo_h = main_bottom_guide - top_guide
        bottom = 1342
        top = bottom - guide_logo_h

        return {
            "main_left": main_left,
            "main_right": main_right,
            "max_left": max_left,
            "max_right": max_right,
            "top": top,
            "bottom": bottom,
            "gradient_darkest": 1360,
        }
    except Exception:
        return fallback

# ── Title splitting ───────────────────────────────────────────────────────────

def split_title(title: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Split title at the first connector word found.
    Returns (line1, connector_word, line2) or (title, None, None).
    """
    words = title.upper().split()
    for i, word in enumerate(words):
        if word in CONNECTOR_WORDS and 0 < i < len(words) - 1:
            return (
                " ".join(words[:i]),
                word,
                " ".join(words[i + 1:]),
            )
    return title.upper(), None, None

# ── MM2K text rendering ───────────────────────────────────────────────────────

def render_text_mm2k(
    img: Image.Image,
    title: str,
    subtitle: Optional[str] = None,
) -> Image.Image:
    """Composite MM2K title text onto a copy of img."""
    result = img.copy()
    draw = ImageDraw.Draw(result)

    # Usable width for MM2K title text.
    # Keep a hard cap of 800px so long single-line titles don't overrun the template.
    usable_w = min(POSTER_W - 2 * BORDER - 80, 800)
    cx = POSTER_W // 2  # centre x

    line1, connector, line2 = split_title(title)

    # Explicit subtitle (e.g. "SEASON ONE") overrides connector-based split
    if subtitle:
        line1 = title.upper()
        connector = None
        line2 = subtitle.upper()

    def draw_line(text: str, y: int, font: ImageFont.FreeTypeFont, alpha: int = 255):
        """Draw centred text — clean white, minimal stroke per MM2K reference."""
        draw.text(
            (cx, y), text, font=font,
            fill=(255, 255, 255, alpha),
            anchor="mm",
            stroke_width=1,
            stroke_fill=(0, 0, 0, 120),
        )

    # Season label is always rendered separately at y=1439 in small font
    # It must not be treated as a second main title line
    SEASON_Y = 1439   # pixel-measured from reference
    TITLE_Y = 1316    # pixel-measured from reference

    if subtitle:
        # Title at y=1316, season label at y=1439 (small, tracked 26px per reference)
        font = fit_font(draw, line1, usable_w, start_size=96)
        draw_line(line1, TITLE_Y, font)
        # Season: 30pt Arial Regular, 26px character tracking, no stroke per reference
        season_font = load_font_regular(30)
        draw_tracked(draw, cx, subtitle, SEASON_Y, season_font, tracking_px=26)
    elif line2 is None:
        # ── Single line: MIDDLE BOTTOM position
        # Reference: text centre at y=1316, 96pt (68px tall)
        font = fit_font(draw, line1, usable_w, start_size=96)
        draw_line(line1, TITLE_Y, font)
    else:
        # ── Title split by connector word (e.g. "GAME OF THRONES")
        longest = line1 if len(line1) >= len(line2) else line2
        font = fit_font(draw, longest, usable_w, start_size=110)
        _, fh = measure_text(draw, longest, font)
        gap = int(fh * 0.25)

        if connector:
            # Three rows: MAIN / ————— OF ————— / SECONDARY centred around y=1316
            connector_str = f"\u2014\u2014\u2014\u2014\u2014 {connector} \u2014\u2014\u2014\u2014\u2014"
            con_size = max(30, int(fh * 0.38))
            small = fit_font(draw, connector_str, usable_w, start_size=con_size)
            _, ch = measure_text(draw, connector_str, small)
            total_h = fh + ch + fh + gap * 2
            top_y = TITLE_Y - total_h // 2 + fh // 2
            draw_line(line1, top_y, font)
            draw_line(connector_str, top_y + fh + gap, small, alpha=210)
            draw_line(line2, top_y + fh + gap + ch + gap, font)
        else:
            # Two lines centred around y=1316
            total_h = fh * 2 + gap
            top_y = TITLE_Y - total_h // 2 + fh // 2
            draw_line(line1, top_y, font)
            draw_line(line2, top_y + fh + gap, font)

    return result

# ── CL2K logo compositing ─────────────────────────────────────────────────────

def render_logo_cl2k(
    img: Image.Image,
    logo: Image.Image,
    season: Optional[int] = None,
    recommended_width: int = 600,
) -> Image.Image:
    """Composite CL2K white logo (and optional season label) onto img.

    Uses the repo guide image as the source of truth for the logo area.
    """
    result = img.copy()
    guide = extract_cl2k_guide_spec()

    logo_rgba = logo.convert("RGBA")

    # Trim transparent padding so sizing uses the real artwork bounds, not the PNG canvas.
    alpha = logo_rgba.getchannel("A")
    bbox = alpha.getbbox()
    if bbox:
        logo_rgba = logo_rgba.crop(bbox)

    lw, lh = logo_rgba.size
    aspect_ratio = lw / lh if lh > 0 else 1.0

    main_w = guide["main_right"] - guide["main_left"]
    max_w = guide["max_right"] - guide["max_left"]
    max_h = guide["bottom"] - guide["top"]

    # Size heuristic for trimmed logos:
    # - stacked / two-line logos need more vertical presence
    # - mid-wide single-line logos can be larger
    # - ultra-wide wordmarks must be smaller or they sprawl too far
    if aspect_ratio < 2.0:
        target_w = 520
        target_h = 150
    elif aspect_ratio < 3.4:
        target_w = 580
        target_h = 145
    elif aspect_ratio < 4.3:
        # Stacked/two-line territory (e.g. The Dark Knight)
        target_w = 620
        target_h = 145
    elif aspect_ratio < 6.5:
        # Mid-wide wordmarks like Ex Machina want presence without overfilling width
        target_w = 700
        target_h = 125
    elif aspect_ratio >= 9.0:
        target_w = 600
        target_h = 95
    else:
        # Interpolate from 700x125 @6.5 down to 600x95 @9.0
        t = (aspect_ratio - 6.5) / 2.5
        target_w = int(round(700 - t * 100))
        target_h = int(round(125 - t * 30))

    allowed_w = min(target_w, max_w)
    allowed_h = min(target_h, max_h)

    scale = min(allowed_w / lw, allowed_h / lh)
    new_w = max(1, int(round(lw * scale)))
    new_h = max(1, int(round(lh * scale)))
    logo_r = logo_rgba.resize((new_w, new_h), Image.LANCZOS)

    # Centre horizontally, bottom-align to the guide-derived baseline,
    # with a small downward nudge to better match reference CL2K posters.
    x = (POSTER_W - new_w) // 2
    y = guide["bottom"] - new_h + 10

    result.paste(logo_r, (x, y), logo_r)

    # Add season label below logo if requested
    if season is not None:
        draw = ImageDraw.Draw(result)
        season_word = SEASON_WORDS.get(season, str(season))
        label = f"SEASON {season_word}"
        cx = POSTER_W // 2
        SEASON_LABEL_Y = 1440  # pixel-measured from reference CL2K poster
        # 30pt Arial Regular, 26px tracking — matches MM2K season label style
        season_font = load_font_regular(30)
        draw_tracked(draw, cx, label, SEASON_LABEL_Y, season_font, tracking_px=26)

    return result

# ── Poster assembly ───────────────────────────────────────────────────────────

def build_base(
    bg: Image.Image,
    title: str = "",
    media_type: str = "movie",
    bg_source: str = "tmdb",
) -> Image.Image:
    """
    Assemble the shared base poster:
      1. Fit background to 1000×1500 using smart crop focus
      2. Black gradient overlay (bottom 40%)
      3. White border (7px)
    """
    focus_x, focus_y = suggest_background_focus(bg, title, media_type, bg_source)
    base = fit_to_poster(bg.convert("RGBA"), focus_x=focus_x, focus_y=focus_y)
    gradient = make_gradient_overlay(POSTER_W, POSTER_H)
    base = Image.alpha_composite(base, gradient)
    base = add_white_border(base)
    return base


def save_jpg(img: Image.Image, path: Path, quality: int = 92):
    img.convert("RGB").save(str(path), format="JPEG", quality=quality, subsampling=0)
    print(f"  ✓ {path}")

# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate MM2K and CL2K Plex media posters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("title", help="Movie or show title")
    parser.add_argument("--year", type=int, help="Release year")
    parser.add_argument(
        "--type", dest="media_type", choices=["movie", "show"], default="movie",
        help="Media type (default: movie)",
    )
    parser.add_argument("--season", type=int, help="Season number (shows only)")
    parser.add_argument(
        "--background-url",
        help="Override background image with a specific URL (uses this instead of Apple/TMDB auto-selection)",
    )
    parser.add_argument(
        "--background-file",
        help="Override background image with a local file path (uses this instead of Apple/TMDB auto-selection)",
    )
    parser.add_argument("--output-dir", default="/home/russell/.openclaw/media/tool-image-generation", help="Output directory (default: OpenClaw media dir)")
    parser.add_argument(
        "--skip-inpaint", action="store_true",
        help="Skip OpenAI inpainting (faster, background may contain text/logos)",
    )
    parser.add_argument(
        "--no-cl2k", action="store_true",
        help="Only generate MM2K poster (skip CL2K)",
    )
    parser.add_argument(
        "--no-mm2k", action="store_true",
        help="Only generate CL2K poster (skip MM2K)",
    )
    parser.add_argument(
        "--no-apple", action="store_true",
        help="Skip Apple TV logo lookup, always use TMDB + Gemini for CL2K",
    )
    parser.add_argument(
        "--apple-storefront", default=APPLE_STOREFRONT,
        help=f"Apple TV storefront ID (default: {APPLE_STOREFRONT} = US)",
    )
    args = parser.parse_args()

    tmdb_key = os.environ.get("TMDB_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if not tmdb_key:
        sys.exit("Error: TMDB_API_KEY is not set. See .env.example.")

    do_inpaint = not args.skip_inpaint
    if do_inpaint and not openai_key:
        print("Warning: OPENAI_API_KEY not set — skipping inpainting step.")
        do_inpaint = False

    # ── Search TMDB ──────────────────────────────────────────────────────────
    print(f"\nSearching TMDB for: {args.title!r} ({args.media_type}) …")
    try:
        result = search_tmdb(args.title, args.year, args.media_type, tmdb_key)
    except Exception as e:
        sys.exit(f"TMDB search failed: {e}")

    media_id = result["id"]
    year = args.year or get_release_year(result, args.media_type)
    title_key = "name" if args.media_type == "show" else "title"
    canonical = result.get(title_key, args.title)
    print(f"  Found: {canonical} ({year})  [TMDB id={media_id}]")

    # ── Fetch images ─────────────────────────────────────────────────────────
    print("  Fetching image metadata …")
    try:
        images = fetch_images(media_id, args.media_type, tmdb_key)
    except Exception as e:
        sys.exit(f"TMDB images fetch failed: {e}")

    bg = None
    bg_source = "tmdb"
    if args.background_file:
        print(f"  Background override file → {args.background_file}")
        try:
            bg = Image.open(args.background_file).convert("RGBA")
            bg_source = "override"
        except Exception as e:
            sys.exit(f"Failed to load background file: {e}")
    elif args.background_url:
        print(f"  Background override URL → {args.background_url}")
        try:
            bg = download_image(args.background_url)
            bg_source = "override"
        except Exception as e:
            sys.exit(f"Failed to download background override: {e}")
    else:
        if not args.no_apple:
            print("  Trying Apple TV background artwork …")
            bg = fetch_apple_tv_background(canonical, args.media_type, args.apple_storefront)
            if bg is not None:
                bg_source = "apple"
                print("  Background → Apple TV CenteredFullScreenBackgroundSmallImage")

        if bg is None:
            bg_url = best_background(images)
            if not bg_url:
                sys.exit("Error: No background image found on Apple TV or TMDB for this title.")
            print(f"  Background → {bg_url}")
            try:
                bg = download_image(bg_url)
            except Exception as e:
                sys.exit(f"Failed to download background: {e}")

    # ── Optional inpainting ───────────────────────────────────────────────────
    if do_inpaint:
        print("  Running OpenAI gpt-image-1 inpainting to remove text …")
        try:
            bg = inpaint_remove_text(bg, openai_key)
            print("  Inpainting complete.")
        except Exception as e:
            print(f"  Warning: Inpainting failed ({e}) — continuing without it.")

    # ── Build shared base ────────────────────────────────────────────────────
    base = build_base(bg, canonical, args.media_type, bg_source)

    # ── Output paths ─────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("  Fetching external IDs …")
    try:
        ext_ids = fetch_external_ids(media_id, args.media_type, tmdb_key)
    except Exception:
        ext_ids = {}
    stem = build_filename_stem(canonical, year, media_id, ext_ids, args.media_type, args.season)

    print()

    # ── MM2K poster ──────────────────────────────────────────────────────────
    if not args.no_mm2k:
        if args.season is not None:
            season_word = SEASON_WORDS.get(args.season, str(args.season))
            subtitle = f"SEASON {season_word}"  # tracking applied at render time
        else:
            subtitle = None
        mm2k = render_text_mm2k(base, canonical, subtitle=subtitle)
        save_jpg(mm2k, out_dir / f"{stem}.jpg")

    # ── CL2K poster ──────────────────────────────────────────────────────────
    if not args.no_cl2k:
        apple_logo = None
        if not args.no_apple:
            print("  Trying Apple TV SingleColorContentLogo …")
            apple_logo = fetch_apple_tv_logo(canonical, args.media_type, args.apple_storefront)

        logo_img = None
        recommended_width = 600
        gemini_key = os.environ.get("GEMINI_API_KEY", "")

        if apple_logo:
            white_score = logo_white_score(apple_logo)
            if white_score >= 0.80:
                print(f"  Using Apple TV logo as-is (white score {white_score:.0%}).")
                logo_img = apple_logo
                lw, lh = apple_logo.size
                aspect_ratio = lw / lh if lh > 0 else 1
                recommended_width = 800 if aspect_ratio > 5.0 else 600
            else:
                if not gemini_key:
                    sys.exit("Error: GEMINI_API_KEY is required to whiten non-white Apple logos. See .env.example.")
                print(f"  Apple logo is dark/coloured (white score {white_score:.0%}) — converting to white via Gemini Flash …")
                logo_img, recommended_width = force_white_logo_gemini(apple_logo, gemini_key)
        else:
            logo_url = best_logo(images)
            if not logo_url:
                print("  No logo found on TMDB or Apple TV — skipping CL2K poster.")
            else:
                print(f"  Logo → {logo_url}")
                try:
                    logo_img = download_image(logo_url)
                    if not gemini_key:
                        sys.exit("Error: GEMINI_API_KEY is required for CL2K logo conversion. See .env.example.")
                    print("  Converting logo to white via Gemini Flash …")
                    logo_img, recommended_width = force_white_logo_gemini(logo_img, gemini_key)
                except Exception as e:
                    print(f"  Warning: CL2K generation failed ({e}).")
                    logo_img = None

        if logo_img is not None:
            cl2k = render_logo_cl2k(base, logo_img, season=args.season, recommended_width=recommended_width)
            save_jpg(cl2k, out_dir / f"{stem}-CL2K.jpg")

    print("\nDone.")


if __name__ == "__main__":
    main()
