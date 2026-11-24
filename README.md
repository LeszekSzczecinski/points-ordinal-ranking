# points-ordinal-ranking

Code for the paper *Ranking by points and ordinal models*: fits of the adjacent-categories (AC)
and cumulative-logit (CL) models to league results, and the scripts that compute the tables and
the numbers of Sec. 4.

## Contents

| Folder | Contents |
|---|---|
| `src/` | model fitting (`optimize_models.py`, `RegularizedRegression.py`), data loading (`manage_data.py`), scores and orders (`verify_score_order.py`) |
| `scripts/` | fits, data download, and one script per table or number of the paper |
| `data/` | match results, not included; see `data/README.md` |
| `results/` | fitted results used in the paper; see `results/README.md` |

## Installation

```
pip install -r requirements.txt
```

The results were computed with Python 3.11.

## What reproduces what

All scripts are run from the repository root. The scripts in the first group read only
`results/`; the others also need the match data of the leagues they cover.

| Paper | Script | Needs |
|---|---|---|
| Table 1; Sec. 4.1, refit with eta = 0 | `python scripts/table1_params.py` | results |
| Sec. 4.1, season-by-season fits | `python scripts/collect_per_season.py` | results |
| Table 2, schedule columns and reordered pairs | `python scripts/table2_schedule.py` | results, data |
| Table 2, columns CL and "both" | `python scripts/cl_table2.py` | results, data |
| Sec. 4.2, AC against CL (log-loss, orders) | `python scripts/ac_vs_cl.py` | results, data |
| Sec. 4.2, score differences of the reordered pairs | `python scripts/sec42_score_gap.py` | results, data (NHL) |
| Sec. 4.2, simulation of the venue effect | `python scripts/sim_venue_reversal.py SHL_4 300 2010 2011 2012 2013 2014` | results, data (SHL) |
| Table 3 | `python scripts/table3_points.py` | results, data |
| Appendix A, counterexample without (S2) | `python scripts/s2_counterexample.py` | nothing |

The files in `results/` are the fitted results used in the paper. To recompute them from the
data:

| Fits | Command |
|---|---|
| all pooled fits (AC, AC with eta = 0, CL, AC with free slopes) | `bash scripts/fit_all.sh` |
| season-by-season fits (NHL, SHL, SuperLega) | `python scripts/run_per_season.py` |

The pooled fits of the NHL and the SHL take hours each.

## Data

The football results of football-data.co.uk can be downloaded with
`python scripts/fetch_football_data.py`. The other results are available from the sources listed
in `data/README.md`, which also gives the file format the code reads.

## Settings

`POR_DATA` and `POR_RESULTS` set the data and results folders (default: `data/` and `results/`).

## License

MIT (see `LICENSE`).
