"""Collect per-season EM fits (AC, symmetric, home boost) and test parameter stability.

For each league: per-season delta_y (free slopes), eta, gamma with model-based SEs
(inverse Hessian of the marginal likelihood), the precision-weighted mean, the
chi-square of the per-season estimates around it (df = T-1), and the pooled fit.
"""
import os
import sys, numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import optimize_models as om
R = om.RESULTS_ROOT
cfg = om.SEASON_CONFIG_JOINT
plan = [('NHL', 'NHL_4'), ('SHL', 'SHL_4'), ('SuperLega', 'SuperLega')]


def load(f):
    d = np.load(f, allow_pickle=True)
    p = d['params']
    pp = p[0] if p.ndim else p.item()
    A = np.asarray(pp['Adelta'], float)
    b = np.asarray(pp['bdelta'], float)
    dvec = np.asarray(pp['delta'], float)
    delta_full = A @ dvec + b
    fv = [str(x) for x in d['free_vars']]
    ks = [int(x) for x in d['key_sizes']]
    cov = np.asarray(d['cov'], float)
    off = np.cumsum([0] + ks)
    sl = {k: slice(off[i], off[i + 1]) for i, k in enumerate(fv)}
    cd = cov[sl['delta'], sl['delta']]
    cov_full = A @ cd @ A.T
    L = len(delta_full)
    free = list(range(1, L // 2))          # free slopes delta_1 .. delta_{floor((L-2)/2)}
    out = {f'delta{y}': (delta_full[y], np.sqrt(cov_full[y, y])) for y in free}
    out['eta'] = (float(np.asarray(pp['eta'])), np.sqrt(cov[sl['eta'], sl['eta']][0, 0]))
    g = float(np.asarray(pp['gamma']))
    out['gamma'] = (g, g * np.sqrt(cov[sl['gamma'], sl['gamma']][0, 0]))   # SE on log gamma -> delta method
    return out


def summary(name, est, se, pooled=None):
    est, se = np.asarray(est), np.asarray(se)
    w = 1 / se**2
    m = np.sum(w * est) / np.sum(w)
    chi2 = np.sum(w * (est - m) ** 2)
    df = len(est) - 1
    p = stats.chi2.sf(chi2, df)
    s = (f'  {name:7s} T={len(est):2d}  range [{est.min():.3f}, {est.max():.3f}]  '
         f'spread SD={est.std(ddof=1):.3f}  median SE={np.median(se):.3f}  '
         f'wmean={m:.3f}+-{np.sqrt(1/np.sum(w)):.3f}  chi2={chi2:.1f} (df={df}, p={p:.3f})')
    if pooled is not None:
        s += f'  pooled={pooled[0]:.3f}+-{pooled[1]:.3f}'
    print(s)


for lg, key in plan:
    print(f'\n=== {lg} ===')
    rows = {}
    for y in cfg[key]:
        f = R / lg / f'AC_{lg}_{y}-{y+1}_sym_em.npz'
        if not f.exists():
            print(f'  missing {y}')
            continue
        rows[y] = load(f)
    if not rows:
        continue
    pf = R / lg / f'AC_{key}_pooled_sym_em.npz'
    pooled = load(pf) if pf.exists() else {}
    keys = list(next(iter(rows.values())).keys())
    print('  season  ' + '  '.join(f'{k:>15s}' for k in keys))
    for y, r in rows.items():
        print(f'  {y}    ' + '  '.join(f'{r[k][0]:7.3f}+-{r[k][1]:5.3f}' for k in keys))
    for k in keys:
        if k == 'gamma':
            continue
        summary(k, [r[k][0] for r in rows.values()], [r[k][1] for r in rows.values()], pooled.get(k))
    g = np.array([r['gamma'][0] for r in rows.values()])
    print(f'  gamma   range [{g.min():.2f}, {g.max():.2f}], median {np.median(g):.2f}'
          + (f'; pooled {pooled["gamma"][0]:.2f}' if 'gamma' in pooled else ''))
