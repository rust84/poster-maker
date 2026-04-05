"""Shared constants for poster_maker."""

from pathlib import Path

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

# Path to the CL2K guide image (shared by ai.py and render.py)
CL2K_GUIDE_PATH = Path(__file__).resolve().parent.parent / "assets" / "cl2k-guide.jpg"
