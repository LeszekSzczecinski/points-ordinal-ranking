"""Table 2 of the paper: per-era schedule structure and score-MAP reorderings, at the fitted eta and with eta=0.
Run (from the repository root): python scripts/table2_schedule.py
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import verify_score_order as v

ERAS = {
    "SHL_4":  [("1999-2003", range(1999, 2004)), ("2010-2014", range(2010, 2015)),
               ("2015-2025", range(2015, 2026))],
    "NHL_4":  [("2005-2007", range(2005, 2008)), ("2008-2011", range(2008, 2012)),
               ("2013-2018", range(2013, 2019)), ("2021-2025", range(2021, 2026))],
}
FLAT = ["EPL", "Championship", "Bundesliga_2pt", "Bundesliga_3pt",
        "Division1_2pt", "Division1_3pt", "SuperLega"]
SPORT = {"SHL_4": "SHL", "NHL_4": "NHL", "EPL": "EPL", "Championship": "Championship",
         "Bundesliga_2pt": "Bundesliga", "Bundesliga_3pt": "Bundesliga",
         "Division1_2pt": "Division1", "Division1_3pt": "Division1",
         "SuperLega": "SuperLega"}


def season_stats(home, away, idx):
    ids = set(idx.tolist())
    m = np.array([(h in ids) for h in home])
    rm = {q: k for k, q in enumerate(idx.tolist())}
    n = len(idx)
    K = np.zeros((n, n), int)
    D = np.zeros((n, n), int)
    for h, a in zip(home[m], away[m]):
        i, j = rm[h], rm[a]
        K[i, j] += 1; K[j, i] += 1; D[i, j] += 1; D[j, i] -= 1
    vb = np.array([bool(np.all(D[i] == 0)) for i in range(n)])
    off = ~np.eye(n, dtype=bool)
    return K, vb, sorted(set(K[off].tolist()))


def league(name, groups=None):
    home, away, Y, names, L = v.load_league(name)
    g = v.season_groups(names)
    fits = {}
    for tag in ("sym_em", "sym_em_eta0"):
        p = np.load(v.RESULTS / SPORT[name] / f"AC_{name}_pooled_{tag}.npz",
                    allow_pickle=True)
        d = np.asarray(p["delta"]).ravel()
        fits[tag] = (np.asarray(p["theta"]).ravel(),
                     v.team_scores(home, away, Y, d))
    if groups is None:
        groups = [(name, [int(y) for y in g])]
    out = []
    for label, yrs in groups:
        yrs = {str(y) for y in yrs} & set(g)
        acc = dict(teams=0, seq=0, cov=0, pairs=0,
                   strict=0, re_h=0, re_0=0, mt=set(), ns=0)
        for yr in sorted(yrs):
            idx = g[yr]
            K, vb, mt = season_stats(home, away, idx)
            n = len(idx)
            acc["ns"] += 1; acc["teams"] += int(vb.sum()); acc["mt"] |= set(mt)
            for i in range(n):
                for j in range(i + 1, n):
                    oth = [l for l in range(n) if l not in (i, j)]
                    se = all(K[i, l] == K[j, l] for l in oth)
                    acc["pairs"] += 1; acc["seq"] += se
                    acc["cov"] += se and vb[i] and vb[j]
                    gi, gj = idx[i], idx[j]
                    for tag, key in (("sym_em", "re_h"), ("sym_em_eta0", "re_0")):
                        th, s = fits[tag]
                        ds, dt = s[gi] - s[gj], th[gi] - th[gj]
                        if abs(ds) < 1e-9:
                            continue
                        if tag == "sym_em":
                            acc["strict"] += 1
                        if ds * dt < 0:
                            acc[key] += 1
            acc["nteam"] = n
        out.append((label, acc))
    return out


rows = []
for nm, grp in ERAS.items():
    rows += [(nm, r) for r in league(nm, grp)]
bal = dict(teams=0, seq=0, cov=0, pairs=0, strict=0, re_h=0, re_0=0, mt=set(), ns=0)
for nm in FLAT:
    for _, a in league(nm):
        for k in ("teams", "seq", "cov", "pairs", "strict", "re_h", "re_0", "ns"):
            bal[k] += a[k]
        bal["mt"] |= a["mt"]
rows.append(("balanced", ("football + SuperLega", bal)))

hdr = f"{'league / era':26s} {'sns':>3s} {'meet':>10s} {'vbal':>9s} {'sch-eq':>11s} {'Prop1':>11s} {'reord':>12s} {'reord eta=0':>12s}"
print(hdr); print("-" * len(hdr))
for nm, (label, a) in rows:
    tt = a["teams"]; nt = a["ns"]
    print(f"{label:26s} {a['ns']:3d} {str(sorted(a['mt'])):>10s} "
          f"{a['teams']:>4d} tms {a['seq']:5d}/{a['pairs']:<5d} {a['cov']:5d}/{a['pairs']:<5d} "
          f"{a['re_h']:4d}/{a['strict']:<6d} {a['re_0']:4d}/{a['strict']:<6d}")
