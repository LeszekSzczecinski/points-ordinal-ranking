"""Download the football results used in the paper from football-data.co.uk.

The site offers one CSV file per league and season. This script downloads the files for
the Premier League (E0), the Championship (E1) and the Bundesliga (D1), seasons 1993/94
to 2025/26, keeps the five columns the code reads (Date, HomeTeam, AwayTeam, FTHG, FTAG),
and writes them to data/<league>/<league>-<year>-<year+1>.csv.

Not covered by this script (see data/README.md): the Premier League 1992/93, the First
Division before 1992 and the Bundesliga before 1993/94 (worldfootball.net), the NHL, the
SHL and SuperLega.

Usage (from the repository root):
    python scripts/fetch_football_data.py                 all leagues and seasons
    python scripts/fetch_football_data.py EPL 2010 2011   one league, selected start years
"""
import csv
import io
import os
import sys
import time
import urllib.request

URL = "https://www.football-data.co.uk/mmz4281/{code}/{div}.csv"
LEAGUES = {"EPL": "E0", "Championship": "E1", "Bundesliga": "D1"}
FIRST, LAST = 1993, 2025
COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
ROOT = os.environ.get("POR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))


def season_code(year):
    """1993 -> '9394', 2023 -> '2324'."""
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def fetch(league, year):
    url = URL.format(code=season_code(year), div=LEAGUES[league])
    req = urllib.request.Request(url, headers={"User-Agent": "points-ordinal-ranking"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    text = raw.decode("utf-8-sig", errors="replace") if raw[:3] == b"\xef\xbb\xbf" else raw.decode("latin-1")
    rows = list(csv.reader(io.StringIO(text)))
    head = [h.strip() for h in rows[0]]
    idx = [head.index(c) for c in COLUMNS]
    out = [COLUMNS]
    for r in rows[1:]:
        if len(r) <= max(idx) or not r[idx[1]].strip():
            continue                                   # trailing empty lines
        out.append([r[i].strip() for i in idx])
    folder = os.path.join(ROOT, league)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{league}-{year}-{year + 1}.csv")
    with open(path, "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(out)
    return path, len(out) - 1


if __name__ == "__main__":
    leagues = [sys.argv[1]] if len(sys.argv) > 1 else list(LEAGUES)
    years = [int(y) for y in sys.argv[2:]] or list(range(FIRST, LAST + 1))
    for lg in leagues:
        for y in years:
            path, n = fetch(lg, y)
            print(f"{lg} {y}-{y + 1}: {n} matches -> {path}")
            time.sleep(1)                              # one request per second
