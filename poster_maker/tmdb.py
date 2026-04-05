"""TMDB API helpers: search, image metadata, filename building, logo/background selection."""

import io
from typing import Optional

import numpy as np
import requests
from PIL import Image

from .config import TMDB_BASE, IMG_BASE


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
    rgba = img.convert("RGBA")
    bbox = rgba.getchannel("A").getbbox()
    if bbox:
        rgba = rgba.crop(bbox)
    arr = np.array(rgba)
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
            thumb_url = f"https://image.tmdb.org/t/p/w300{item['file_path']}"
            r = requests.get(thumb_url, timeout=10)
            img = Image.open(io.BytesIO(r.content)).convert("RGBA")
            return logo_white_score(img)
        except Exception:
            return 0.0

    print(f"  Checking {len(candidates)} logos for whiteness …")
    scored = [(logo_whiteness(l), l.get("vote_average", 0), l) for l in candidates]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_white_frac, _, best = scored[0]
    print(f"  Best logo whiteness: {best_white_frac:.0%} | {best['file_path']}")
    return IMG_BASE + best["file_path"]
