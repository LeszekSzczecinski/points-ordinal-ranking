# Table 2 of the paper, columns CL and "both".
# For each row of Table 2: pairs with distinct scores, reordered by the AC MAP
# skills (must reproduce Table 2), reordered by the CL MAP skills (the new
# column), and pairs ordered differently by the AC and CL skills.
# Both fits: joint EM, home boost, ridge prior (results/<league>/{AC,CL}_*_pooled_sym_em.npz).
# Run (from the repository root): python scripts/cl_table2.py [CL_DIR_OVERRIDE]
import os
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import optimize_models as om              # noqa: E402
import verify_score_order as vso          # noqa: E402

CL_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else None   # e.g. a folder of refitted CL results
ROWS = [
    ("football, SuperLega", [(n, None) for n in ["EPL", "Championship", "Division1_2pt", "Division1_3pt",
                                                 "Bundesliga_2pt", "Bundesliga_3pt", "SuperLega"]]),
    ("SHL 2015-2025", [("SHL_4", range(2015, 2026))]),
    ("SHL 2010-2014", [("SHL_4", range(2010, 2015))]),
    ("SHL 1999-2003", [("SHL_4", range(1999, 2004))]),
    ("NHL 2005-2007", [("NHL_4", range(2005, 2008))]),
    ("NHL 2008-2011", [("NHL_4", range(2008, 2012))]),
    ("NHL 2013-2018", [("NHL_4", range(2013, 2019))]),
    ("NHL 2021-2025", [("NHL_4", range(2021, 2026))]),
]
TOL = 1e-6
cache = {}


def league(name):
    if name in cache:
        return cache[name]
    sport = om._base_sport(name)
    ac = np.load(om.RESULTS_ROOT / sport / f"AC_{name}_pooled_sym_em.npz", allow_pickle=True)
    clp = (CL_DIR / f"CL_{name}_pooled_sym_em.npz") if CL_DIR and (CL_DIR / f"CL_{name}_pooled_sym_em.npz").exists() \
        else om.RESULTS_ROOT / sport / f"CL_{name}_pooled_sym_em.npz"
    cl = np.load(clp, allow_pickle=True)
    home, away, Y, names, L = vso.load_league(name)
    ok = [str(x) for x in ac["names"]] == names == [str(x) for x in cl["names"]]
    s = vso.team_scores(home, away, Y, ac["delta"].flatten())
    cache[name] = (ok, names, s, ac["theta"].flatten(), cl["theta"].flatten(), str(clp))
    return cache[name]


print(f"{'row':20s} {'distinct':>8s} {'AC reord':>9s} {'CL reord':>9s} {'AC!=CL':>8s} {'all pairs':>9s} {'AC!=CL distinct':>16s} {'both':>5s}")
for label, parts in ROWS:
    d = ra = rc = dis = tot = disd = both = 0
    bad = []
    for name, years in parts:
        ok, names, s, ta, tc, clp = league(name)
        if not ok:
            bad.append(name); continue
        for yr, idx in vso.season_groups(names).items():
            if years is not None and int(yr) not in years:
                continue
            n = len(idx)
            for i in range(n):
                for j in range(i + 1, n):
                    gi, gj = idx[i], idx[j]
                    ds, da, dc = s[gi] - s[gj], ta[gi] - ta[gj], tc[gi] - tc[gj]
                    tot += 1
                    if abs(da) > TOL and abs(dc) > TOL and da * dc < 0:
                        dis += 1
                    if abs(ds) > 1e-9:
                        d += 1; ra += int(ds * da < 0); rc += int(ds * dc < 0)
                        disd += int(abs(da) > TOL and abs(dc) > TOL and da * dc < 0)
                        both += int(ds * da < 0 and ds * dc < 0)   # Table 2 column "both"
    note = f"  (CL fit stale, skipped: {', '.join(bad)})" if bad else ""
    if d == 0:
        print(f"{label:20s} {'-':>8s}{note}"); continue
    print(f"{label:20s} {d:8d} {ra:9d} {rc:4d} ({rc / d:5.1%}) {dis:4d} ({dis / tot:5.1%}) {tot:6d} {disd:6d} ({disd / d:5.1%}) {both:5d}{note}")
