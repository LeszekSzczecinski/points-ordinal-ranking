# Sec. 4.2 of the paper: AC against CL, predictive log-loss (ALO) and orders.
# Compares, per league, the AC fit (constant-sum slopes, home boost) with the
# CL (cumulative-logit) fit with the same home boost and prior, both stored in
# results/<league>/: predictive score (ALO log-loss per match) and the MAP orders
# (pairs within a season ordered differently by AC and CL), split by whether
# the season's schedule is round-robin and venue-balanced.  Also the score
# order against each model's MAP order.
# Run (from the repository root): python scripts/ac_vs_cl.py
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import optimize_models as om              # noqa: E402
import verify_score_order as vso          # noqa: E402

TOL = 1e-6
configs = ["EPL", "Championship", "Division1_2pt", "Division1_3pt", "Bundesliga_2pt",
           "Bundesliga_3pt", "NHL_4", "SHL_4", "SuperLega"]
print(f"{'league':15s} {'ALO_AC':>8s} {'ALO_CL':>8s} {'CL-AC':>8s} | {'pairs':>6s} {'AC!=CL':>7s} {'bal':>5s} {'bal!=':>5s} | {'s!=AC':>6s} {'s!=CL':>6s} {'s!=CL bal':>9s}")
for name in configs:
    sport = om._base_sport(name)
    ac = np.load(om.RESULTS_ROOT / sport / f"AC_{name}_pooled_sym_em.npz", allow_pickle=True)
    cl = np.load(om.RESULTS_ROOT / sport / f"CL_{name}_pooled_sym_em.npz", allow_pickle=True)
    a_alo = float(ac["alo"].item()["full"][0]); c_alo = float(cl["alo"].item()["full"][0])
    home, away, Y, names, L = vso.load_league(name)
    if not ([str(x) for x in ac["names"]] == names == [str(x) for x in cl["names"]]):
        print(f"{name:15s} NAME MISMATCH: AC {len(ac['names'])}, CL {len(cl['names'])}, data {len(names)} teams; skipped")
        continue
    ta, tc = ac["theta"].flatten(), cl["theta"].flatten()
    s = vso.team_scores(home, away, Y, ac["delta"].flatten())
    tot = dis = bal = baldis = sa = sc = scb = 0
    for yr, idx in vso.season_groups(names).items():
        sb, vb = vso.season_balance(home, away, idx, Y)
        b = sb and vb
        n = len(idx)
        for i in range(n):
            for j in range(i + 1, n):
                gi, gj = idx[i], idx[j]
                da, dc, ds = ta[gi] - ta[gj], tc[gi] - tc[gj], s[gi] - s[gj]
                tot += 1; bal += int(b)
                if abs(da) > TOL and abs(dc) > TOL and da * dc < 0:
                    dis += 1; baldis += int(b)
                if abs(ds) > 1e-9:
                    sa += int(ds * da < 0); sc += int(ds * dc < 0); scb += int(b and ds * dc < 0)
    print(f"{name:15s} {a_alo:8.5f} {c_alo:8.5f} {c_alo - a_alo:+8.5f} | {tot:6d} {dis:7d} {bal:5d} {baldis:5d} | {sa:6d} {sc:6d} {scb:9d}")
