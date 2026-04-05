"""CLI entry point for poster generation."""

import argparse
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from PIL import Image

from .config import APPLE_STOREFRONT, SEASON_WORDS
from .tmdb import (
    search_tmdb, fetch_images, fetch_external_ids,
    build_filename_stem, best_background, best_logo, logo_white_score,
)
from .apple import (
    download_image, get_release_year,
    fetch_apple_tv_background, fetch_apple_tv_logo,
)
from .ai import force_white_logo_gemini, inpaint_remove_text
from .render import build_base, render_text_mm2k, render_logo_cl2k, save_jpg


def main():
    parser = argparse.ArgumentParser(
        description="Generate MM2K and CL2K Plex media posters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
    python poster.py "Inception"
    python poster.py "The Dark Knight" --year 2008
    python poster.py "Breaking Bad" --type show --season 1
""",
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
    parser.add_argument(
        "--output-dir",
        default="/home/russell/.openclaw/media/tool-image-generation",
        help="Output directory (default: OpenClaw media dir)",
    )
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
            print(f"\n  [!] Inpainting FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            print("  [!] Poster will use the unedited background — source text/logos may be visible.", file=sys.stderr)
            print("  [!] Re-run with --skip-inpaint to suppress this warning.\n", file=sys.stderr)

    # ── Build shared base ────────────────────────────────────────────────────
    base = build_base(bg)

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
