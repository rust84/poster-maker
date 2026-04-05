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

# Words that trigger a line split in MM2K titles
CONNECTOR_WORDS = {"OF", "AND", "VS", "IN", "AT", "FOR"}

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
            import io as _io, numpy as _np
            url = IMG_BASE + item["file_path"]
            # Use a small thumbnail from TMDB to check colour
            thumb_url = f"https://image.tmdb.org/t/p/w300{item['file_path']}"
            r = requests.get(thumb_url, timeout=10)
            img = Image.open(_io.BytesIO(r.content)).convert("RGBA")
            arr = _np.array(img)
            visible = arr[:,:,3] > 10
            if visible.sum() == 0:
                return 0.0
            r_ch, g_ch, b_ch = arr[visible,0].astype(float), arr[visible,1].astype(float), arr[visible,2].astype(float)
            maxc = _np.maximum(_np.maximum(r_ch, g_ch), b_ch)
            minc = _np.minimum(_np.minimum(r_ch, g_ch), b_ch)
            sat = _np.where(maxc > 0, (maxc - minc) / maxc, 0.0)
            brightness = (r_ch + g_ch + b_ch) / 3.0
            # Score: fraction of pixels that are low-sat AND bright (white)
            white_frac = float(((sat < 0.15) & (brightness > 200)).mean())
            return white_frac
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

    # Load guidelines screenshot to give Gemini context on sizing
    guidelines_path = "/home/russell/.openclaw/media/inbound/file_1---d9387b2d-0745-48ff-b417-214855229e41.jpg"
    try:
        with open(guidelines_path, "rb") as gf:
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

def fit_to_poster(img: Image.Image) -> Image.Image:
    """Scale and centre-crop image to exactly POSTER_W × POSTER_H."""
    tw, th = POSTER_W, POSTER_H
    iw, ih = img.size
    scale = max(tw / iw, th / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return resized.crop((left, top, left + tw, top + th))


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

    # Usable width: full width minus border (7px each side) minus 40px padding each side
    usable_w = POSTER_W - 2 * BORDER - 80
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

    if line2 is None:
        # ── Single line: MIDDLE BOTTOM position
        # Reference measurement: text centre at y=1316, font size 96pt (68px tall)
        # Scale down only if text is too wide for usable area
        font = fit_font(draw, line1, usable_w, start_size=96)
        draw_line(line1, 1316, font)
    else:
        # ── Two lines
        # Fit to the longest line
        longest = line1 if len(line1) >= len(line2) else line2
        font = fit_font(draw, longest, usable_w, start_size=110)
        _, fh = measure_text(draw, longest, font)
        gap = int(fh * 0.25)  # space between lines

        if connector:
            # Three rows: MAIN / ————— OF ————— / SECONDARY
            connector_str = f"\u2014\u2014\u2014\u2014\u2014 {connector} \u2014\u2014\u2014\u2014\u2014"
            con_size = max(30, int(fh * 0.38))
            small = fit_font(draw, connector_str, usable_w, start_size=con_size)
            _, ch = measure_text(draw, connector_str, small)
            # Stack: line1, connector, line2 centred around y=1380
            total_h = fh + ch + fh + gap * 2
            top_y = 1380 - total_h // 2 + fh // 2
            draw_line(line1, top_y, font)
            draw_line(connector_str, top_y + fh + gap, small, alpha=210)
            draw_line(line2, top_y + fh + gap + ch + gap, font)
        else:
            # Two lines centred around y=1380
            total_h = fh * 2 + gap
            top_y = 1380 - total_h // 2 + fh // 2
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
    """Composite CL2K white logo (and optional season label) onto img."""
    result = img.copy()

    # Resize logo: 600px default per spec, 800px only for very wide logos (AR > 5:1)
    lw, lh = logo.size
    aspect_ratio = lw / lh if lh > 0 else 1
    if recommended_width >= 800 and aspect_ratio > 5.0:
        target_w = 800
    else:
        target_w = 600
    scale = target_w / lw
    new_w = int(lw * scale)
    new_h = int(lh * scale)
    logo_r = logo.convert("RGBA").resize((new_w, new_h), Image.LANCZOS)

    # Centre horizontally, bottom-align to 'Main Logo Bottom' guideline (y=1342 per spec)
    # Pixel-measured from reference CL2K poster
    LOGO_BOTTOM = 1342
    x = (POSTER_W - new_w) // 2
    y = LOGO_BOTTOM - new_h

    result.paste(logo_r, (x, y), logo_r)

    # Add season label below logo if requested
    if season is not None:
        draw = ImageDraw.Draw(result)
        label = f"SEASON {season}"
        cx = POSTER_W // 2
        label_y = y + new_h + 40
        font = fit_font(draw, label, POSTER_W - 2 * BORDER - 80, start_size=60)
        draw.text(
            (cx, label_y), label, font=font,
            fill=(255, 255, 255, 220),
            anchor="mm",
            stroke_width=2,
            stroke_fill=(0, 0, 0, 160),
        )

    return result

# ── Poster assembly ───────────────────────────────────────────────────────────

def build_base(bg: Image.Image) -> Image.Image:
    """
    Assemble the shared base poster:
      1. Fit background to 1000×1500
      2. Black gradient overlay (bottom 40%)
      3. White border (7px)
    """
    base = fit_to_poster(bg.convert("RGBA"))
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

    bg_url = best_background(images)
    if not bg_url:
        sys.exit("Error: No background image found on TMDB for this title.")

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
    base = build_base(bg)

    # ── Output paths ─────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    year_str = f" ({year})" if year else ""
    safe = canonical.replace("/", "-").replace(":", " -")
    stem = f"{safe} - Season {args.season}{year_str}" if args.season else f"{safe}{year_str}"

    print()

    # ── MM2K poster ──────────────────────────────────────────────────────────
    if not args.no_mm2k:
        subtitle = f"Season {args.season}" if args.season else None
        mm2k = render_text_mm2k(base, canonical, subtitle=subtitle)
        save_jpg(mm2k, out_dir / f"{stem}.jpg")

    # ── CL2K poster ──────────────────────────────────────────────────────────
    if not args.no_cl2k:
        logo_url = best_logo(images)
        if not logo_url:
            print("  No logo found on TMDB — skipping CL2K poster.")
        else:
            print(f"  Logo → {logo_url}")
            try:
                logo_img = download_image(logo_url)
                gemini_key = os.environ.get("GEMINI_API_KEY", "")
                recommended_width = 600
                if not gemini_key:
                    sys.exit("Error: GEMINI_API_KEY is required for CL2K logo conversion. See .env.example.")
                print("  Converting logo to white via Gemini Flash …")
                logo_img, recommended_width = force_white_logo_gemini(logo_img, gemini_key)
                cl2k = render_logo_cl2k(base, logo_img, season=args.season, recommended_width=recommended_width)
                save_jpg(cl2k, out_dir / f"{stem}-CL2K.jpg")
            except Exception as e:
                print(f"  Warning: CL2K generation failed ({e}).")

    print("\nDone.")


if __name__ == "__main__":
    main()
