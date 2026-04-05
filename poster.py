#!/usr/bin/env python3
"""poster.py — Generate MM2K and CL2K style Plex media posters.

Usage:
    python poster.py "Inception"
    python poster.py "The Dark Knight" --year 2008
    python poster.py "Breaking Bad" --type show --season 1
"""

from poster_maker.cli import main

if __name__ == "__main__":
    main()
