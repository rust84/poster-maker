"""AI-assisted image processing: Gemini logo whitening and OpenAI inpainting."""

import base64
import io
import json

import numpy as np
import requests
from PIL import Image, ImageDraw

from .config import CL2K_GUIDE_PATH


def force_white_logo_gemini(logo: Image.Image, api_key: str):
    """Use Gemini Flash vision to intelligently extract logo as clean white on transparent bg.

    Returns (Image, recommended_width_px).
    """
    from google import genai as google_genai
    from google.genai import types as gtypes

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
        model="gemini-2.0-flash",
        contents=contents,
    )

    text = response.text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        info = json.loads(text)
    except Exception:
        # Fallback: use safe conservative thresholds rather than calling removed function
        print("  Gemini response parse failed, using fallback thresholds")
        info = {
            "has_transparency": False,
            "bg_is_coloured": True,
            "text_is_light": False,
            "saturation_threshold": 0.20,
            "brightness_threshold": 150,
            "recommended_width_px": 600,
        }

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
