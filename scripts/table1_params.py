# Table 1 (tab:params) of main.tex: free slopes delta_y and eta with model-based SEs
# (inverse curvature of the marginal likelihood), from the pooled AC fits with the
# home boost. Also the Sec. 4.1 "not shown" (ii): the shift of each free slope when the
# league is refitted with eta=0, in standard errors of the eta-hat fit.
# Inputs: results/<sport>/AC_<name>_pooled_sym_em.npz and ..._pooled_sym_em_eta0.npz
# Run (from the repository root): python scripts/table1_params.py
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import optimize_models as om              # noqa: E402

NAMES = ["Division1_2pt", "Division1_3pt", "EPL", "Championship", "Bundesliga_2pt",
         "Bundesliga_3pt", "NHL_4", "SHL_4", "SuperLega"]


def load(f):
    """Free slopes and eta with SEs from a saved EM fit (same as collect_per_season.py)."""
    d = np.load(f, allow_pickle=True)
    p = d["params"]
    pp = p[0] if p.ndim else p.item()
    A = np.asarray(pp["Adelta"], float)
    b = np.asarray(pp["bdelta"], float)
    delta_full = A @ np.asarray(pp["delta"], float) + b
    fv = [str(x) for x in d["free_vars"]]
    ks = [int(x) for x in d["key_sizes"]]
    cov = np.asarray(d["cov"], float)
    off = np.cumsum([0] + ks)
    sl = {k: slice(off[i], off[i + 1]) for i, k in enumerate(fv)}
    out = {}
    L = len(delta_full)
    if "delta" in sl:
        cov_full = A @ cov[sl["delta"], sl["delta"]] @ A.T
        for y in range(1, L // 2):
            out[f"delta{y}"] = (delta_full[y], np.sqrt(cov_full[y, y]))
    else:
        for y in range(1, L // 2):
            out[f"delta{y}"] = (delta_full[y], np.nan)
    if "eta" in sl:
        out["eta"] = (float(np.asarray(pp["eta"])), np.sqrt(cov[sl["eta"], sl["eta"]][0, 0]))
    return out


print(f"{'league':16s} {'seasons':>14s}  free slopes (eta-hat fit)        eta")
shift = []
for name in NAMES:
    sport = om._base_sport(name)
    f = om.RESULTS_ROOT / sport / f"AC_{name}_pooled_sym_em.npz"
    r = load(f)
    yrs = om.SEASON_CONFIG_JOINT[name]
    slopes = ", ".join(f"{r[k][0]:.2f}+-{r[k][1]:.2f}" for k in r if k.startswith("delta")) or "none"
    print(f"{name:16s} {min(yrs)}-{max(yrs)} ({len(yrs):2d})  {slopes:30s}  {r['eta'][0]:.2f}+-{r['eta'][1]:.2f}")
    f0 = om.RESULTS_ROOT / sport / f"AC_{name}_pooled_sym_em_eta0.npz"
    if f0.exists():
        r0 = load(f0)
        for k in r:
            if k.startswith("delta"):
                z = abs(r0[k][0] - r[k][0]) / r[k][1]
                shift.append((name, k, r[k][0], r0[k][0], z))

print("\nSec. 4.1 (ii): free-slope shift under the eta=0 refit, in SEs of the eta-hat fit")
for name, k, d1, d0, z in shift:
    print(f"  {name:10s} {k}: {d1:.4f} -> {d0:.4f}   |shift|/SE = {z:.2f}")
print(f"  max |shift|/SE = {max(z for *_, z in shift):.2f}")
