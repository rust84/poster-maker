# poster-maker

Automated generator for community-standard MM2K and CL2K style Plex media posters.

Follows the [daps poster creation guidelines](https://github.com/Drazzilb08/daps/wiki/how-to-create-posters-the-right-way) to the letter.

## Styles

### MM2K (MusikMann2000)
- 1000×1500px, white 25px border
- Textless background from TMDB
- Black gradient overlay (y=1065 → y=1360)
- ALL CAPS title in **Arial Bold**, centred at y=1316
- Exported as `.jpg` quality 92

### CL2K (Clear Logo 2K)
- Same base as MM2K
- Official show/movie **white logo** instead of text
- Logo: 600px wide (800px max for very wide logos), bottom-aligned to y=1342
- Logo source: TMDB logos (whitest available selected automatically)
- Gemini Flash used to intelligently extract white logo from coloured originals

## Requirements

```bash
pip install -r requirements.txt
```

## Setup

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

| Key | Required | Notes |
|-----|----------|-------|
| `TMDB_API_KEY` | ✅ | Free at [themoviedb.org](https://www.themoviedb.org/settings/api) |
| `OPENAI_API_KEY` | Optional | For background text removal (inpainting). Skip with `--skip-inpaint` |
| `GEMINI_API_KEY` | Optional | For intelligent logo-to-white conversion. Falls back to PIL extraction |

## Usage

```bash
# Movie poster (both MM2K and CL2K)
python poster.py "Inception"
python poster.py "The Dark Knight" --year 2008

# TV show (with season)
python poster.py "Breaking Bad" --type show --season 1

# Skip OpenAI inpainting (faster, background may have text)
python poster.py "Inception" --skip-inpaint

# Only generate one style
python poster.py "Inception" --no-cl2k
python poster.py "Inception" --no-mm2k

# Custom output directory
python poster.py "Inception" --output-dir ~/posters
```

## Output

Files are named using Trash-Guides / TPDB convention:

```
Inception (2010).jpg          # MM2K
Inception (2010)-CL2K.jpg     # CL2K
Breaking Bad - Season 1 (2008).jpg
Breaking Bad - Season 1 (2008)-CL2K.jpg
```

## Font

Requires **Arial Bold**. Install to `~/.local/share/fonts/Arial_Bold.ttf` on Linux.
Falls back to Liberation Sans Bold (metric-compatible) if Arial is not available.
