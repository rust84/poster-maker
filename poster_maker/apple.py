"""Apple TV artwork integration: background images and single-colour logos."""

import io
from typing import Optional

import requests
from PIL import Image

from .config import APPLE_STOREFRONT, APPLE_URL_SERVICE


def download_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGBA")


def get_release_year(result: dict, media_type: str) -> Optional[int]:
    key = "first_air_date" if media_type == "show" else "release_date"
    date = result.get(key, "") or ""
    return int(date[:4]) if len(date) >= 4 and date[:4].isdigit() else None


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
