# Sec. 4.2 of main.tex, "not shown": in the NHL, the median score difference of the
# reordered pairs against that of all pairs with distinct scores. Score = fitted slopes
# (constant-sum), skills = pooled MAP theta of the AC fit with the home boost; pairs
# within a season.
# Run (from the repository root): python scripts/sec42_score_gap.py
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import optimize_models as om              # noqa: E402
import verify_score_order as vso          # noqa: E402

name = "NHL_4"
r = np.load(om.RESULTS_ROOT / "NHL" / f"AC_{name}_pooled_sym_em.npz", allow_pickle=True)
home, away, Y, names, L = vso.load_league(name)
assert [str(x) for x in r["names"]] == names
theta = r["theta"].flatten()
delta = r["delta"].flatten()
s = vso.team_scores(home, away, Y, delta)
gap_all, gap_inv = [], []
for yr, idx in vso.season_groups(names).items():
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            ds = s[i] - s[j]
            if abs(ds) <= vso.TOL:
                continue
            gap_all.append(abs(ds))
            if ds * (theta[i] - theta[j]) < 0:
                gap_inv.append(abs(ds))
print(f"delta = {np.round(delta, 4)}")
print(f"pairs with distinct scores: {len(gap_all)}, reordered: {len(gap_inv)}")
print(f"median |score difference|: reordered {np.median(gap_inv):.2f}, all {np.median(gap_all):.2f}")
