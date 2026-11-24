#!/usr/bin/env python3
"""
verify_score_order.py  --  auditable score-vs-MAP ranking comparison.

Replaces the earlier ad-hoc, unsaved pass that produced the section-4 tables.
Every number here rests on one consistent rule set so it can be inspected.

WHAT IT CHECKS
--------------
For each pooled league fit (SEASON_CONFIG_JOINT in optimize_models), per season:

  1. SCHEDULE AXIS  (reproduces "Ranking discrepancies", Table A)
     Rank teams by the symmetric-model score s_i and by the symmetric-model
     MAP skill theta_i.  Count strict-order inversions: pairs with DISTINCT
     score (|Delta s| > tol) whose theta order disagrees.  On a schedule- and
     venue-balanced season this MUST be 0 (Proposition 1 / Corollary).

  2. SELF-TEST  (Kendall tau of the same two orders)
     On a balanced season the symmetric score order and the symmetric theta
     order are identical INCLUDING ties, so Kendall tau-b MUST equal 1.0.
     Any tau < 1 on a balanced season signals a bug, not tie-breaking.

  3. WEIGHTS AXIS  (reproduces "Point systems and the skill order", Table B)
     Same reference theta (symmetric MAP).  Compare, by Kendall tau per season:
       col A: model score-points (symmetric delta) order  vs theta
       col B: league ACTUAL points order                  vs theta
     Report the paired difference Delta tau = tau_A - tau_B and the count of
     seasons with tau_A >= tau_B.  Everything is within ONE model (symmetric),
     so col A is the self-test (=1 on balanced) and col B isolates the effect
     of the published weights alone.

KEY CONVENTIONS (from optimize_models.make_regression_data)
    Outcome ordering: Y = 0  = away win (worst for home)
                      Y = L-1 = home win (best for home)
    Home team's own outcome = Y ;  away team's own outcome = L-1-Y.

SCORE RULE (Lemma "Constant-sum of the score")
    Each game distributes a fixed total of 1 between the two teams:
        s[home] += delta[Y]
        s[away] += 1 - delta[Y]
    For the symmetric model delta[Y] = 1 - delta[L-1-Y], so a team's score is
    its own-outcome weight regardless of venue.  Using a single draw value for
    both venues (the earlier bug) is only correct for the symmetric model; the
    rule above is correct for any delta.

Run (no fitting here):
    python src/verify_score_order.py
"""

import os
import sys
import copy
import numpy as np
from scipy.stats import kendalltau

import optimize_models as om

RESULTS = om.RESULTS_ROOT
TOL = 1e-9          # score-tie tolerance: below this two scores are "equal"
# Both the score and theta carry ~1e-15 float noise on values that are
# analytically equal (e.g. two balanced teams with an exact score tie whose
# skills are equal, or a summation-order difference of 3.5e-15 in the score).
# kendalltau would read such a noise-level difference as a one-sided tie and
# drag tau-b below 1.  Round both to ROUND_DEC decimals first: this sits far
# above the noise (<=1e-15) and far below the smallest GENUINE gap in either
# variable (score and theta gaps among distinct pairs are all >= ~6e-5), so no
# distinct teams are merged.  The strict-inversion count needs no rounding -- it
# only inspects distinct-score pairs, whose theta gaps are real.
ROUND_DEC = 9


def tau(x, theta):
    """Kendall tau-b with both arguments de-noised so float-level differences
    on analytically-equal values do not count as one-sided ties."""
    return kendalltau(np.round(x, ROUND_DEC), np.round(theta, ROUND_DEC)).statistic

# League actual points, indexed by a team's OWN outcome 0..L-1 (worst..best).
ACTUAL_POINTS = {
    "Division1_2pt":  [0, 1, 2],
    "Division1_3pt":  [0, 1, 3],
    "EPL":            [0, 1, 3],
    "Championship":   [0, 1, 3],
    "Bundesliga_2pt": [0, 1, 2],
    "Bundesliga_3pt": [0, 1, 3],
    # NHL: reg loss / OT loss / OT win / reg win  ->  0 / 1 / 2 / 2
    "NHL_4":          [0, 1, 2, 2],
    # SHL: 0 / 1 / 2 / 3
    "SHL_4":          [0, 1, 2, 3],
    # SuperLega: 0-3, 1-3, 2-3, 3-2, 3-1, 3-0  ->  0 / 0 / 1 / 2 / 3 / 3
    "SuperLega":      [0, 0, 1, 2, 3, 3],
}


def load_league(name):
    """Return (home, away, Y, names, L) for a pooled config, via the SAME
    loader the fits used, so theta/names alignment is guaranteed."""
    years = om.SEASON_CONFIG_JOINT[name]
    sport = om._base_sport(name)
    result_cat = om.RESULT_CAT_JOINT.get(name, None)
    (_, y_super, u), names, L = om._make_super_season(sport, years, result_cat)
    home = np.asarray(u["home_idx"], dtype=int)
    away = np.asarray(u["away_idx"], dtype=int)
    Y = np.asarray(y_super, dtype=int)
    return home, away, Y, list(names), L


def team_scores(home, away, Y, delta):
    """Constant-sum score: s[h] += delta[Y]; s[a] += 1 - delta[Y]."""
    M = int(max(home.max(), away.max())) + 1
    s = np.zeros(M)
    np.add.at(s, home, delta[Y])
    np.add.at(s, away, 1.0 - delta[Y])
    return s


def team_points(home, away, Y, P, L):
    """League actual points by own outcome (home own=Y, away own=L-1-Y)."""
    M = int(max(home.max(), away.max())) + 1
    P = np.asarray(P, dtype=float)
    pts = np.zeros(M)
    np.add.at(pts, home, P[Y])
    np.add.at(pts, away, P[L - 1 - Y])
    return pts


def season_groups(names):
    """year -> array of global team indices (names are 'Team_year')."""
    groups = {}
    for i, nm in enumerate(names):
        yr = nm.rsplit("_", 1)[-1]
        groups.setdefault(yr, []).append(i)
    return {yr: np.array(ix) for yr, ix in groups.items()}


def season_balance(home, away, idx, Y):
    """(schedule_balanced, venue_balanced) for the teams in one season."""
    idx_set = set(idx.tolist())
    mask = np.array([(h in idx_set) for h in home])
    hh, aa = home[mask], away[mask]
    remap = {g: k for k, g in enumerate(idx.tolist())}
    n = len(idx)
    K = np.zeros((n, n), dtype=int)     # total meetings
    Dv = np.zeros((n, n), dtype=int)    # home_i - away_i signed venue count
    for h, a in zip(hh, aa):
        i, j = remap[h], remap[a]
        K[i, j] += 1; K[j, i] += 1
        Dv[i, j] += 1; Dv[j, i] -= 1
    off = ~np.eye(n, dtype=bool)
    sched_bal = len(np.unique(K[off])) == 1
    venue_bal = np.all(Dv == 0)
    return sched_bal, venue_bal


def strict_inversions(s, theta, tol=TOL):
    """Count pairs with distinct score whose theta order disagrees."""
    n = len(s)
    inv = 0
    strict = 0
    for i in range(n):
        for j in range(i + 1, n):
            ds = s[i] - s[j]
            if abs(ds) <= tol:
                continue
            strict += 1
            if ds * (theta[i] - theta[j]) < 0:
                inv += 1
    return inv, strict


def run():
    configs = ["EPL", "Championship", "Division1_2pt", "Division1_3pt",
               "Bundesliga_2pt", "Bundesliga_3pt", "NHL_4", "SHL_4",
               "SuperLega"]

    print("=" * 78)
    print("SCHEDULE AXIS + SELF-TEST  (symmetric model: score vs symmetric MAP theta)")
    print("=" * 78)
    print(f"{'League':16s} {'seasons':>7s} {'bal':>4s} {'strict':>7s} "
          f"{'inv':>4s} {'rate':>8s} {'minTau':>7s} {'bal.minTau':>10s}")

    weights_rows = []
    for name in configs:
        sym_path = RESULTS / om._base_sport(name) / f"AC_{name}_pooled_sym_em.npz"
        if not sym_path.exists():
            print(f"{name:16s}  MISSING {sym_path}")
            continue
        r = np.load(sym_path, allow_pickle=True)
        theta = r["theta"].flatten()
        delta = r["delta"].flatten()
        npz_names = [str(x) for x in r["names"]]

        home, away, Y, names, L = load_league(name)
        if names != npz_names:
            print(f"{name:16s}  NAME MISMATCH ({len(names)} vs {len(npz_names)}) -- skipping")
            continue

        s_model = team_scores(home, away, Y, delta)
        P = ACTUAL_POINTS[name]
        pts_actual = team_points(home, away, Y, P, L)

        groups = season_groups(names)
        tot_inv = tot_strict = 0
        n_bal = 0
        taus = []
        bal_taus = []
        dtau_list = []
        model_better = 0
        n_seasons = 0
        for yr, idx in groups.items():
            si = s_model[idx]; ti = theta[idx]; pi = pts_actual[idx]
            inv, strict = strict_inversions(si, ti)
            tot_inv += inv; tot_strict += strict
            sb, vb = season_balance(home, away, idx, Y)
            balanced = sb and vb
            n_bal += int(balanced)
            tau_model = tau(si, ti)
            taus.append(tau_model)
            if balanced:
                bal_taus.append(tau_model)
            tau_actual = tau(pi, ti)
            dtau_list.append(tau_model - tau_actual)
            model_better += int((tau_model - tau_actual) > 1e-12)
            n_seasons += 1

        rate = tot_inv / tot_strict if tot_strict else 0.0
        min_tau = min(taus)
        min_bal_tau = min(bal_taus) if bal_taus else float("nan")
        print(f"{name:16s} {n_seasons:7d} {n_bal:4d} {tot_strict:7d} "
              f"{tot_inv:4d} {rate:8.4%} {min_tau:7.4f} {min_bal_tau:10.4f}")

        weights_rows.append((name, n_seasons,
                             float(np.mean(taus)),
                             float(np.mean([tau(pts_actual[idx], theta[idx])
                                            for idx in groups.values()])),
                             float(np.mean(dtau_list)),
                             float(np.std(dtau_list)),
                             model_better))

    print()
    print("=" * 78)
    print("WEIGHTS AXIS  (reference = symmetric MAP theta)")
    print("  tau_model = model score-points order vs theta  (SELF-TEST: 1.0 on balanced)")
    print("  tau_act   = league actual-points order vs theta")
    print("  d_tau     = paired (tau_model - tau_act), mean (std) over seasons")
    print("=" * 78)
    print(f"{'League':16s} {'n':>3s} {'tau_model':>9s} {'tau_act':>8s} "
          f"{'d_tau':>7s} {'(std)':>7s} {'modelBetter':>11s}")
    for name, n, tm, ta, dt, ds, mb in weights_rows:
        print(f"{name:16s} {n:3d} {tm:9.4f} {ta:8.4f} {dt:7.4f} {ds:7.4f} "
              f"{mb:6d}/{n:<4d}")


def run_cl():
    """CL departure from the score on balanced schedules.

    The point-counting score (AC symmetric delta = the standings) is sufficient
    only for the AC model.  The CL model is a legitimate single-scalar-skill
    model but the score is NOT sufficient for it (its gradient does not separate
    into delta_y - G(z)), so ranking by CL-fitted theta should reorder strictly
    score-ordered pairs EVEN ON A BALANCED SCHEDULE, where AC gives exactly 0.
    This is the empirical counterpart of the Sec-2 AC-vs-CL sufficiency contrast.
    Reference score = AC symmetric delta; reference theta = CL sym_em fit.
    """
    configs = ["EPL", "Championship", "Division1_2pt", "Division1_3pt",
               "Bundesliga_2pt", "Bundesliga_3pt", "NHL_4", "SHL_4",
               "SuperLega"]
    print()
    print("=" * 78)
    print("CL DEPARTURE  (point-counting score vs CL-fitted theta)")
    print("  AC gives 0 inversions on balanced seasons; CL should NOT, because")
    print("  the score is not a sufficient statistic for the CL model.")
    print("=" * 78)
    print(f"{'League':16s} {'seasons':>7s} {'bal':>4s} {'strict':>7s} "
          f"{'inv':>4s} {'rate':>8s} {'balInv':>6s} {'medGap':>7s}")
    for name in configs:
        ac_path = RESULTS / om._base_sport(name) / f"AC_{name}_pooled_sym_em.npz"
        cl_path = RESULTS / om._base_sport(name) / f"CL_{name}_pooled_sym_em.npz"
        if not cl_path.exists():
            print(f"{name:16s}  MISSING {cl_path.name}")
            continue
        ac = np.load(ac_path, allow_pickle=True)
        cl = np.load(cl_path, allow_pickle=True)
        delta = ac["delta"].flatten()
        cl_theta = cl["theta"].flatten()
        cl_names = [str(x) for x in cl["names"]]

        home, away, Y, names, L = load_league(name)
        if names != cl_names:
            print(f"{name:16s}  NAME MISMATCH (CL) -- skipping")
            continue

        s_model = team_scores(home, away, Y, delta)
        groups = season_groups(names)
        tot_inv = tot_strict = bal_inv = n_bal = 0
        gaps = []
        for yr, idx in groups.items():
            si = s_model[idx]; ti = cl_theta[idx]
            sb, vb = season_balance(home, away, idx, Y)
            balanced = sb and vb
            n_bal += int(balanced)
            n = len(idx)
            for i in range(n):
                for j in range(i + 1, n):
                    ds = si[i] - si[j]
                    if abs(ds) <= TOL:
                        continue
                    tot_strict += 1
                    if ds * (ti[i] - ti[j]) < 0:
                        tot_inv += 1
                        gaps.append(abs(ti[i] - ti[j]))
                        if balanced:
                            bal_inv += 1
        rate = tot_inv / tot_strict if tot_strict else 0.0
        med = float(np.median(gaps)) if gaps else float("nan")
        print(f"{name:16s} {len(groups):7d} {n_bal:4d} {tot_strict:7d} "
              f"{tot_inv:4d} {rate:8.4%} {bal_inv:6d} {med:7.4f}")


def run_eta():
    """Venue isolation: MAP order with eta_hat vs eta=0, toggling ONLY eta.

    Direct control for the "meetings not venue" claim in Sec 4.  Starting from the
    fitted eta_hat parameters, we hold delta, alpha and gamma FIXED and recompute
    the MAP skills with eta forced to 0 (theta_hat).  The only thing that changes
    between the two orders is the home advantage, so any reordering is purely the
    venue effect on the given schedule.  (Comparing the two SEPARATE eta_hat and
    eta=0 fits instead would also move delta for L>3 and conflate the two effects;
    that is why an eta=0 REFIT reorders balanced SuperLega/NHL seasons where this
    clean toggle does not.)  On a fully balanced season both orders equal the score
    order (Corollary), so the count MUST be 0 there.  Gaps near 0 = near-ties only.
    """
    fitter = om.RR.make_fitter("AC")
    configs = ["EPL", "Championship", "Division1_2pt", "Division1_3pt",
               "Bundesliga_2pt", "Bundesliga_3pt", "NHL_4", "SHL_4",
               "SuperLega"]
    print()
    print("=" * 78)
    print("VENUE ISOLATION  (MAP eta_hat  vs  MAP eta=0; toggle ONLY eta)")
    print("  fully-balanced season MUST give 0 reorderings (both = score order);")
    print("  unbalanced schedules show how much the home advantage moves the order.")
    print("=" * 78)
    print(f"{'League':16s} {'seasons':>7s} {'full':>4s} {'eta_hat':>7s} "
          f"{'strict':>7s} {'disc':>4s} {'rate':>8s} {'meanTau':>8s} {'medGap':>7s}")
    for name in configs:
        eta_path = RESULTS / om._base_sport(name) / f"AC_{name}_pooled_sym_em.npz"
        if not eta_path.exists():
            print(f"{name:16s}  MISSING {eta_path.name}")
            continue
        re = np.load(eta_path, allow_pickle=True)
        years = om.SEASON_CONFIG_JOINT[name]
        sport = om._base_sport(name)
        result_cat = om.RESULT_CAT_JOINT.get(name, None)
        (X, y_super, u), names, L = om._make_super_season(sport, years, result_cat)
        data = (X, y_super, u)
        home = np.asarray(u["home_idx"], dtype=int)
        away = np.asarray(u["away_idx"], dtype=int)
        Y = np.asarray(y_super, dtype=int)
        names = list(names)

        pp = copy.deepcopy(re["params"][0])          # eta_hat params
        th_e = np.asarray(fitter.theta_hat(pp, data))
        pp0 = copy.deepcopy(pp); pp0["eta"] = 0.0     # toggle ONLY eta
        th_0 = np.asarray(fitter.theta_hat(pp0, data))

        groups = season_groups(names)
        taus = []
        tot_disc = tot_strict = n_full = 0
        gaps = []
        for yr, idx in groups.items():
            a = th_e[idx]; b = th_0[idx]
            sb, vb = season_balance(home, away, idx, Y)
            n_full += int(sb and vb)
            taus.append(tau(a, b))            # rounds both, tie-robust
            n = len(idx)
            for i in range(n):
                for j in range(i + 1, n):
                    da = a[i] - a[j]; db = b[i] - b[j]
                    if abs(da) <= TOL or abs(db) <= TOL:
                        continue
                    tot_strict += 1
                    if da * db < 0:
                        tot_disc += 1
                        gaps.append(abs(da))
        rate = tot_disc / tot_strict if tot_strict else 0.0
        med = float(np.median(gaps)) if gaps else float("nan")
        eta_hat = float(pp["eta"])
        print(f"{name:16s} {len(groups):7d} {n_full:4d} {eta_hat:7.3f} "
              f"{tot_strict:7d} {tot_disc:4d} {rate:8.4%} "
              f"{np.mean(taus):8.4f} {med:7.4f}")


if __name__ == "__main__":
    run()
    run_cl()
    run_eta()
