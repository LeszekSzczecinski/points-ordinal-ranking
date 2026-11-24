# Sec. 4.2 of the paper: when does venue imbalance reverse the score order?
#
# Parametric simulation from a fitted pooled model (AC, constant-sum slopes,
# home boost, ridge prior).  Kept from the data: the schedule and the fitted
# hyper-parameters (delta, alpha, gamma).  Simulated: the skills, drawn from the
# prior N(0, 1/gamma), and the outcomes, drawn from the model at a home advantage
# eta on a grid.  Each replicate refits the MAP skills with the generating
# parameters and compares them with the model's score (slopes = score-points).
#
# Pairs are split into
#   EQ : schedule-equivalent, not venue-balanced for both players.  By Prop. 1
#        nothing can be reordered at eta = 0, so every reordering is a venue effect.
#        For them the exact identity from eq. (MAP.split) holds:
#            s_i - s_j = [term with the sign of z_ij] + V_ij,
#            V_ij = sum_{l!=i} d_il Gc(z_il) - sum_{l!=j} d_jl Gc(z_jl),
#        so a reversal happens iff sign(s_i-s_j) = sign(V_ij) and |s_i-s_j| < |V_ij|.
#   NE : not schedule-equivalent.  At eta = 0 only the meeting counts reorder
#        them; the increase with eta is the contribution of the venue.
#
# Run (from the repository root): python scripts/sim_venue_reversal.py LEAGUE N_REP [YEAR ...]
#      e.g.  SHL_4 300 2010 2011 2012 2013 2014     or     NHL_4 200

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import optimize_models as om  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "SHL_4"
N_REP = int(sys.argv[2]) if len(sys.argv) > 2 else 50
YEARS = tuple(sys.argv[3:])            # empty = all seasons of the pooled fit
ETAS = (0.0, 0.2, 0.42, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0)
TOL_S = 1e-9        # score tie
TOL_T = 1e-5        # skill tie (BFGS noise is far below this)
SEED = 20261102


def load():
    years = om.SEASON_CONFIG_JOINT[NAME]
    sport = om._base_sport(NAME)
    (X, y, u), names, L = om._make_super_season(sport, years, om.RESULT_CAT_JOINT.get(NAME, None))
    r = np.load(om.RESULTS_ROOT / sport / f"AC_{NAME}_pooled_sym_em.npz", allow_pickle=True)
    assert [str(x) for x in r["names"]] == list(names)
    return X, y, u, list(names), L, r["params"][0], r["delta"].flatten()


def season_structure(home, away, names):
    """Per season: team indices, venue matrix D, pair arrays (I, J) and the
    EQ / NE masks (see header)."""
    all_years = sorted({nm.rsplit("_", 1)[-1] for nm in names})
    out = []
    for yr in (YEARS or all_years):
        idx = np.array([i for i, nm in enumerate(names) if nm.rsplit("_", 1)[-1] == yr])
        pos = {g: k for k, g in enumerate(idx)}
        n = len(idx)
        K = np.zeros((n, n), int)
        D = np.zeros((n, n), int)
        for h, a in zip(home, away):
            if h in pos:
                i, j = pos[h], pos[a]
                K[i, j] += 1; K[j, i] += 1
                D[i, j] += 1; D[j, i] -= 1
        vb = np.all(D == 0, axis=1)
        I, J = np.triu_indices(n, 1)
        eq = np.zeros(len(I), bool)
        for p, (i, j) in enumerate(zip(I, J)):
            others = np.ones(n, bool); others[[i, j]] = False
            eq[p] = np.all(K[i, others] == K[j, others])
        EQ = eq & ~(vb[I] & vb[J])
        NE = ~eq
        # richness of the venue imbalance on EQ pairs: sum_l |d_il - d_jl| over l != i,j
        rich = np.array([np.abs(np.delete(D[i] - D[j], [i, j])).sum() for i, j in zip(I, J)])
        out.append(dict(idx=idx, D=D, I=I, J=J, EQ=EQ, NE=NE, rich=rich, hh=D[I, J]))
    return out


def scores(home, away, Y, delta, M):
    s = np.zeros(M)
    np.add.at(s, home, delta[Y])
    np.add.at(s, away, 1.0 - delta[Y])
    return s


def main():
    X, y, u, names, L, pp0, delta = load()
    fitter = om.RR.make_fitter("AC")
    home = np.asarray(u["home_idx"], int)
    away = np.asarray(u["away_idx"], int)
    hfa = np.asarray(u["hfa"], float)
    M = X.shape[0]
    gamma = float(np.ravel(pp0["gamma"])[0])
    seasons = season_structure(home, away, names)
    nEQ = sum(int(S["EQ"].sum()) for S in seasons)
    nNE = sum(int(S["NE"].sum()) for S in seasons)
    rich = np.concatenate([S["rich"][S["EQ"]] for S in seasons])
    hh = np.concatenate([np.abs(S["hh"][S["EQ"]]) for S in seasons])
    print(f"{NAME} seasons {YEARS[0] + '-' + YEARS[-1] if YEARS else 'all'}: EQ pairs {nEQ}, NE pairs {nNE}; "
          f"gamma={gamma:.3f}, delta={np.round(delta, 3)}, fitted eta={float(np.ravel(pp0['eta'])[0]):.3f}")
    if nEQ:
        vals, cnt = np.unique(rich, return_counts=True)
        print("  EQ pairs, sum_l |d_il - d_jl| (l != i,j):", dict(zip(vals.tolist(), cnt.tolist())),
              "; |d_ij|:", dict(zip(*[a.tolist() for a in np.unique(hh, return_counts=True)])))

    rng = np.random.default_rng(SEED)
    print(f"{'eta':>5s} {'rep':>4s} | {'EQ dist':>8s} {'EQ rev':>7s} {'rate':>7s} {'EQ tied':>7s} {'broken':>6s} "
          f"{'not_at_risk':>11s} | {'NE dist':>8s} {'NE rev':>7s} {'rate':>7s}")
    for eta in ETAS:
        pp = dict(pp0); pp["eta"] = np.asarray(eta, dtype=np.asarray(pp0["eta"]).dtype)

        def G_home(z):
            P = np.asarray(fitter.predict_proba(pp, z, hfa=np.ones_like(z)))
            return P @ delta

        def Gc(z):
            return 0.5 * (G_home(z) - 1.0 + G_home(-z))

        c = dict(eqd=0, eqr=0, eqt=0, eqb=0, bad=0, ned=0, ner=0)
        for rep in range(N_REP):
            theta = rng.normal(0.0, 1.0 / np.sqrt(gamma), M)
            z = theta[home] - theta[away]
            P = np.asarray(fitter.predict_proba(pp, z, hfa=hfa))
            cum = (P / P.sum(axis=1, keepdims=True)).cumsum(axis=1)
            Y = (rng.random(len(z))[:, None] > cum).sum(axis=1).clip(0, L - 1)
            th = np.asarray(fitter.theta_hat(pp, (X, np.asarray(Y, dtype=np.asarray(y).dtype), u)))
            s = scores(home, away, Y, delta, M)
            for S in seasons:
                t, ss, I, J = th[S["idx"]], s[S["idx"]], S["I"], S["J"]
                ds, dz = ss[I] - ss[J], t[I] - t[J]
                tied = np.abs(ds) <= TOL_S
                rev = (~tied) & (ds * dz < 0)
                ne = S["NE"]
                c["ned"] += int((ne & ~tied).sum()); c["ner"] += int((ne & rev).sum())
                eq = S["EQ"]
                if eq.any():
                    n = len(t)
                    C = Gc((t[:, None] - t[None, :]).ravel()).reshape(n, n)
                    np.fill_diagonal(C, 0.0)
                    Vrow = (S["D"] * C).sum(axis=1)
                    V = Vrow[I] - Vrow[J]
                    risk = (np.sign(ds) == np.sign(V)) & (np.abs(ds) < np.abs(V))
                    c["eqd"] += int((eq & ~tied).sum()); c["eqr"] += int((eq & rev).sum())
                    c["eqt"] += int((eq & tied).sum()); c["eqb"] += int((eq & tied & (np.abs(dz) > TOL_T)).sum())
                    c["bad"] += int((eq & rev & ~risk).sum())
        r1 = c["eqr"] / c["eqd"] if c["eqd"] else float("nan")
        r2 = c["ner"] / c["ned"] if c["ned"] else float("nan")
        print(f"{eta:5.2f} {N_REP:4d} | {c['eqd']:8d} {c['eqr']:7d} {r1:7.3%} {c['eqt']:7d} {c['eqb']:6d} "
              f"{c['bad']:11d} | {c['ned']:8d} {c['ner']:7d} {r2:7.3%}", flush=True)


if __name__ == "__main__":
    main()
