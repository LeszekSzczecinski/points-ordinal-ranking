# Data

The match results are not included in this repository. Each league is read from
`data/<league>/<league>-<year>-<year+1>.csv`, where `<year>` is the year in which the season starts.
Another location can be set with the environment variable `POR_DATA`.

## Football (downloadable with a script)

`python scripts/fetch_football_data.py` downloads the files offered by
[football-data.co.uk](https://www.football-data.co.uk) for the Premier League (`EPL`),
the Championship (`Championship`) and the Bundesliga (`Bundesliga`), seasons 1993/94 to 2025/26,
and keeps the columns used by the code:

```
Date,HomeTeam,AwayTeam,FTHG,FTAG
14/08/93,Arsenal,Coventry,0,3
```

`FTHG` and `FTAG` are the goals of the home and the away team.

## Other leagues (available from the sources below)

| Folder | League | Seasons | Source |
|---|---|---|---|
| `Division1` | English First Division | 1950/51-1991/92 | worldfootball.net |
| `EPL` | Premier League | 1992/93 | worldfootball.net |
| `Bundesliga_legacy` | Bundesliga | 1965/66-1992/93 | worldfootball.net |
| `NHL` | National Hockey League | 2005/06-2019/20 | hockey-reference.com |
| `NHL` | National Hockey League | 2020/21-2025/26 | nhl.com |
| `SHL` | Swedish Hockey League | 1999/00-2025/26 | stats.swehockey.se |
| `SuperLega` | Italian volleyball, first division | 2009/10-2024/25 | legavolley.it |

The football files use the format above (`Division1` adds a column `round`, which is dropped).
The other formats are:

**NHL** (regular season only). Two formats are read. The schedule-and-results table of
hockey-reference.com, saved as CSV, where the unnamed sixth column is `OT` for a game decided in
overtime, `SO` for a shootout, and empty otherwise (the last three columns are dropped):
```
Date,Visitor,G,Home,G,,Att.,LOG,Notes
2010-10-07,Chicago Blackhawks,3,Colorado Avalanche,4,OT,"18,007",,
```
or, as for the SHL below, `date,home_team,home_score,away_team,away_score,SOOT`.

**SHL** (regular season only; `SOOT` is `OT` for overtime, `SO` for a shootout, empty otherwise)
```
date,home_team,home_score,away_team,away_score,SOOT
2022-09-17,Rogle,5,Brynas,4,OT
```

**SuperLega** (regular season only; the scores are the sets won, so one of them is 3)
```
date,home_team,home_score,away_team,away_score
2010-10-23 10:00:00,Piacenza,0,Monza,3
```

Team names must be consistent within a season; the code does not use them across seasons.
The seasons excluded from the fits (for example those affected by COVID-19) are listed in
`SEASON_CONFIG_JOINT` in `src/optimize_models.py`.
