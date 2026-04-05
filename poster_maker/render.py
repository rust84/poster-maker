"""Image utilities, font helpers, and poster rendering (MM2K + CL2K)."""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import (
    POSTER_W, POSTER_H, BORDER,
    GRADIENT_START_Y, GRADIENT_DARKEST_Y,
    CONNECTOR_WORDS, SEASON_WORDS,
    CL2K_GUIDE_PATH,
)

# ── Font helpers ───────────────────────────────────────────────────────────────

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

_FONT_PATHS_REGULAR = [
    os.path.expanduser("~/.local/share/fonts/Arial.ttf"),
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    # Absolute last resort — tiny bitmap font, won't scale
    return ImageFont.load_default()


def load_font_regular(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_PATHS_REGULAR:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return load_font(size)  # fall back to bold if regular not found


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

# ── CL2K guide ─────────────────────────────────────────────────────────────────

_cl2k_guide_cache: Optional[dict] = None


def extract_cl2k_guide_spec(path: Path = CL2K_GUIDE_PATH) -> dict:
    """Read the CL2K guide image and derive the usable logo guide box.

    The guide image is a 1000px-wide crop of the poster bottom area, so x coordinates
    map directly to poster space. For y, we align the guide's 'Gradient Darkest Line'
    to the known poster y=1360 and derive the other rows from that offset.
    """
    global _cl2k_guide_cache
    if _cl2k_guide_cache is not None:
        return _cl2k_guide_cache

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

        _cl2k_guide_cache = {
            "main_left": main_left,
            "main_right": main_right,
            "max_left": max_left,
            "max_right": max_right,
            "top": top,
            "bottom": bottom,
            "gradient_darkest": 1360,
        }
        return _cl2k_guide_cache
    except Exception:
        _cl2k_guide_cache = fallback
        return fallback

# ── Image utilities ────────────────────────────────────────────────────────────

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
    """Draw solid 25px white border on all sides per MM2K spec."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    # Draw a solid filled rectangle frame (more reliable than outline loop)
    draw.rectangle([0, 0, w, BORDER - 1], fill=(255, 255, 255, 255))        # Top
    draw.rectangle([0, h - BORDER, w, h], fill=(255, 255, 255, 255))        # Bottom
    draw.rectangle([0, 0, BORDER - 1, h], fill=(255, 255, 255, 255))        # Left
    draw.rectangle([w - BORDER, 0, w, h], fill=(255, 255, 255, 255))        # Right
    return img

# ── Title splitting ────────────────────────────────────────────────────────────

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

# ── MM2K text rendering ────────────────────────────────────────────────────────

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

# ── CL2K logo compositing ──────────────────────────────────────────────────────

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

# ── Poster assembly ────────────────────────────────────────────────────────────

def build_base(bg: Image.Image) -> Image.Image:
    """
    Assemble the shared base poster:
      1. Fit background to 1000×1500 (centered crop)
      2. Black gradient overlay (bottom 40%)
      3. White border (25px)
    """
    base = fit_to_poster(bg.convert("RGBA"))
    gradient = make_gradient_overlay(POSTER_W, POSTER_H)
    base = Image.alpha_composite(base, gradient)
    base = add_white_border(base)
    return base


def save_jpg(img: Image.Image, path: Path, quality: int = 92):
    img.convert("RGB").save(str(path), format="JPEG", quality=quality, subsampling=0)
    print(f"  ✓ {path}")
