# Table 3 (tab:points) of main.tex: Kendall tau_b between the skill order (pooled MAP
# theta of the AC fit with the home boost) and the order by (a) the league's own rule,
# (b) the uniform rule xi_y = y/(L-1), (c) the fitted slopes; averaged over the seasons
# of each row, with the SHL split by era.
# Inputs: results/<sport>/AC_<name>_pooled_sym_em.npz and the match data (via the loader
# of verify_score_order.py, the one used by the fits).
# Run (from the repository root): python scripts/table3_points.py
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import optimize_models as om              # noqa: E402
import verify_score_order as vso          # noqa: E402

ROWS = [
    ("Division1 (0-1-2)", "Division1_2pt", None),
    ("Division1 (0-1-3)", "Division1_3pt", None),
    ("EPL (0-1-3)", "EPL", None),
    ("Championship (0-1-3)", "Championship", None),
    ("Bundesliga (0-1-2)", "Bundesliga_2pt", None),
    ("Bundesliga (0-1-3)", "Bundesliga_3pt", None),
    ("NHL (0-1-2-2)", "NHL_4", None),
    ("SHL 1999-2003", "SHL_4", range(1999, 2004)),
    ("SHL 2010-2014", "SHL_4", range(2010, 2015)),
    ("SHL 2015-2025", "SHL_4", range(2015, 2026)),
    ("SuperLega", "SuperLega", None),
]
cache = {}
print(f"{'row':22s} {'n':>3s} {'own':>7s} {'uniform':>8s} {'fitted':>7s}")
for label, name, years in ROWS:
    if name not in cache:
        sport = om._base_sport(name)
        r = np.load(om.RESULTS_ROOT / sport / f"AC_{name}_pooled_sym_em.npz", allow_pickle=True)
        home, away, Y, names, L = vso.load_league(name)
        assert [str(x) for x in r["names"]] == names
        theta = r["theta"].flatten()
        s_fit = vso.team_scores(home, away, Y, r["delta"].flatten())
        s_uni = vso.team_scores(home, away, Y, np.arange(L) / (L - 1))
        s_own = vso.team_points(home, away, Y, vso.ACTUAL_POINTS[name], L)
        cache[name] = (theta, s_own, s_uni, s_fit, vso.season_groups(names))
    theta, s_own, s_uni, s_fit, groups = cache[name]
    t = []
    for yr, idx in groups.items():
        if years is not None and int(yr) not in years:
            continue
        t.append([vso.tau(s[idx], theta[idx]) for s in (s_own, s_uni, s_fit)])
    t = np.mean(t, axis=0)
    print(f"{label:22s} {len(groups) if years is None else len([y for y in groups if int(y) in years]):3d} "
          f"{t[0]:7.3f} {t[1]:8.3f} {t[2]:7.3f}")
