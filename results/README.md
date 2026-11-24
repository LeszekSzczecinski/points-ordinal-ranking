# Fitted results

The `.npz` files in this folder are the output of the fits used in the paper. They are
included so that the tables and the numbers of the paper can be checked without refitting,
which takes hours for the NHL and the SHL. They are not raw data: each file holds the fitted
parameters (slopes, intercepts, home advantage, prior precision), their covariance, and the
fitted skill of each team in each season.

To reproduce these files, the fits must be run again from the match data:

1. obtain the data (see `data/README.md`);
2. run `bash scripts/fit_all.sh` for the pooled fits and `python scripts/run_per_season.py`
   for the season-by-season fits.

The fits overwrite the files in this folder (or in the folder set by `POR_RESULTS`).

## Files

| File | Fit |
|---|---|
| `<sport>/AC_<name>_pooled_sym_em.npz` | AC model, constant-sum slopes, home boost, all seasons of `<name>` |
| `<sport>/AC_<name>_pooled_sym_em_eta0.npz` | the same with the home advantage fixed at 0 |
| `<sport>/CL_<name>_pooled_sym_em.npz` | cumulative-logit model, same home boost and prior |
| `<sport>/AC_<name>_pooled_free_em.npz` | AC model with free slopes |
| `<sport>/AC_<sport>_<year>-<year+1>_sym_em.npz` | AC model fitted to one season (NHL, SHL, SuperLega) |

`<name>` is a configuration of `SEASON_CONFIG_JOINT` in `src/optimize_models.py`
(e.g. `NHL_4`, `Bundesliga_2pt`). Team-season names have the form `Team_year`.

The files are read with `numpy.load(path, allow_pickle=True)`; the entries `params` and `alo`
are Python dictionaries of NumPy arrays.
