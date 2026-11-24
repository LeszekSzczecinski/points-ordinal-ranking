"""
optimize_models.py
==================
Fit ordinal regression models (AC or CL) with L2 regularisation to
single-season or pooled multi-season data for EPL, Championship, Bundesliga,
NHL, SHL, and SuperLega.

Models
------
AC (Adjusting Categories / softmax)
  P(Y=y | z) = softmax(alpha + delta * z)[y]
  Parameters: alpha (outcome intercepts), delta (score weights),
              eta (HFA), gamma (L2 strength).

CL (Cumulative Logistic / proportional-odds)
  P(Y=y | z) = logistic(c_full[y] - z) - logistic(c_full[y-1] - z)
  Parameters: c (free thresholds, strictly increasing), eta (HFA),
              gamma (L2 strength).

CLp (Cumulative Probit)
  Same as CL but with the standard normal CDF instead of logistic.

Both models share the same fitting infrastructure (EM / ML / ALO) and output
structure, differing only in parameter names and model-specific summary
quantities (xi_y for AC; c_full thresholds for CL).

Usage
-----
  # Single season
  res = fit_season("EPL", 2023, model="AC",  method="em")
  res = fit_season("EPL", 2023, model="CL",  method="em")   # logistic
  res = fit_season("EPL", 2023, model="CLp", method="em")   # probit

  # All seasons separately
  fit_all_seasons_separately("EPL", range(1992, 2025), model="CL", method="em")

  # Joint fit across all seasons
  fit_all_seasons_jointly("EPL", range(1992, 2025), model="CL", method="em", n_boot=0)

Normalised objective (RegularizedRegression.py)
------------------------------------------------
  J(theta) = sum_t ell_t  +  (gamma/2) ||theta||^2

  The unnormalised objective gives the correct Laplace approximation:
  H = hessian(J) is the exact Hessian of the negative log-joint.
  Typical optimal gamma is O(T/M) for all supported leagues.

Convention (manage_data)
------------------------
  X has shape M x T; column t has +1 at home-team row and -1 at away-team row.
  z > 0  =>  home team is stronger.
  Outcome ordering: y=0 = away win (worst for home), y=L-1 = home win.

Dependencies
------------
  src/RegularizedRegression.py   JAX fitting engine
  src/manage_data.py             data loading and result coding

Output
------
  results/{sport}/AC_{sport}_{year}-{year+1}_{tag}.npz   (AC model)
  results/{sport}/CL_{sport}_{year}-{year+1}_{tag}.npz   (CL model)
"""

import os
import sys
import re
import copy
import gc
import io
import contextlib
import pathlib
import argparse
import time
import socket

# Limit XLA/Eigen to 1 thread per process.  Our matrices are tiny (M <= ~35
# teams) so multi-threading buys nothing, but multiple jobs on the same node
# would otherwise each try to grab all available CPUs.  setdefault means an
# external OMP_NUM_THREADS= override still takes effect if needed.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import numpy as np
import scipy.linalg
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)   # needed for stable BFGS convergence

# ── path setup ────────────────────────────────────────────────────────────────
_THIS_DIR = pathlib.Path(__file__).resolve().parent          # .../Regression/
_SKF_DIR  = _THIS_DIR.parent / "SKF"

for _p in [str(_THIS_DIR), str(_SKF_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import RegularizedRegression as RR

try:
    import manage_data
    _MANAGE_DATA_AVAILABLE = True
except ImportError:
    manage_data = None
    _MANAGE_DATA_AVAILABLE = False

RESULTS_ROOT = pathlib.Path(os.environ.get("POR_RESULTS", _THIS_DIR.parent / "results"))

# ── season ranges ─────────────────────────────────────────────────────────────
SEASON_CONFIG: dict[str, list[int]] = {
    # English Football League First Division, full history (1950-51 to 1991-92).
    # Note: 3-point system introduced in 1981-82 (year=1981); pre-1981 uses 2 points.
    "Division1":    list(range(1950, 1992)),
    # Premier League, from the inaugural 1992-93 season onward.
    "EPL":          list(range(1992, 2026)),
    "Championship": list(range(1993, 2026)),
    # manage_data auto-redirects Bundesliga -> Bundesliga_legacy for year < 1993,
    # so a single key covers the full history (1965-66 is the first 18-team season).
    # The 3 overlapping seasons (1993-95) exist in both sources with different team
    # name encodings; here we use the main Bundesliga source from 1993 onwards.
    "Bundesliga":   list(range(1965, 2026)),
    # 2004-05 NHL season was cancelled (lockout) -- skip it
    "NHL":          list(range(1996, 2004)) + list(range(2005, 2026)),
    "SHL":          list(range(1993, 2026)),
    "SuperLega":    list(range(2009, 2025)),
}

# ── joint-fit season ranges (era-separated) ───────────────────────────────────
SEASON_CONFIG_JOINT: dict[str, list[int]] = {
    # English First Division -- 2-pt era (1950-51 to 1980-81)
    "Division1_2pt":  list(range(1950, 1981)),
    # English First Division -- 3-pt era (1981-82 to 1991-92)
    "Division1_3pt":  list(range(1981, 1992)),
    # Premier League -- excluding COVID-affected seasons (2019-20 and 2020-21)
    "EPL":            [y for y in range(1992, 2026) if y not in (2019, 2020)],
    # Excluding 2019-20 and 2020-21 (behind closed doors due to Covid-19)
    "Championship":   [y for y in range(1993, 2026) if y not in (2019, 2020)],
    # Bundesliga -- 2-pt era (1965-66 to 1994-95)
    # Excluding 1991-92: one-time 20-team expansion after German reunification.
    "Bundesliga_2pt": [y for y in range(1965, 1995) if y != 1991],
    # Bundesliga -- 3-pt era (1995-96 onward)
    # Excluding 2019-20 and 2020-21 (ghost-games seasons, no spectators).
    "Bundesliga_3pt": [y for y in range(1995, 2026) if y not in (2019, 2020)],
    # NHL -- post-shootout era (2005-06 onward): L=4
    # Excluding: 2012 (lockout-shortened), 2019 (Covid-shortened), 2020 (Covid).
    "NHL_4":          [y for y in range(2005, 2026) if y not in (2012, 2019, 2020)],
    # SHL -- all-OT-decisive era: L=4 (no ties possible)
    # Excluding 2019-20 and 2020-21 (Covid: playoffs cancelled / ghost-games).
    "SHL_4":          list(range(1999, 2004)) + [y for y in range(2010, 2026) if y not in (2019, 2020)],
    # Excluding 2019-20 (truncated) and 2020-21 (Covid).
    "SuperLega":      [y for y in range(2009, 2025) if y not in (2019, 2020)],
}


def _base_sport(name: str) -> str:
    """Strip era suffix to get the actual data-directory name.

    Handles:  _2pt, _3pt  (scoring-era splits for football)
              _3, _4, _5  (outcome-count splits for NHL / SHL)
    E.g.  'Division1_2pt' -> 'Division1',  'NHL_4' -> 'NHL',  'SHL_5' -> 'SHL'.
    """
    return re.sub(r'_\d+(?:pt)?$', '', name)


def _cl_cdf(model: str) -> int:
    """Return the CDF flag for a CL-family model string.

    "CL"  -> 1  (logistic / proportional-odds)
    "CLp" -> 0  (probit)
    """
    if model == "CL":
        return 1
    if model == "CLp":
        return 0
    raise ValueError(f"_cl_cdf called with non-CL model {model!r}")


# Default gamma grid (log-spaced, 15 values from 0.01 to 10)
DEFAULT_GAMMA = np.logspace(-2, 1, 15)

# Per-sport result coding for single-season fits (fit_all_seasons_separately).
RESULT_CAT_DEFAULT: dict[str, int] = {
    "SuperLega": 6,    # 3-0, 3-1, 3-2 W and their mirror losses
}

# Per-entry result coding for joint fits (fit_all_seasons_jointly).
RESULT_CAT_JOINT: dict[str, int] = {
    "NHL_4":     4,    # post-shootout: reg W / OT-SO W / OT-SO L / reg L
    "SHL_4":     4,    # all-OT-decisive era (1999-2003 golden-goal + 2010+)
    "SuperLega": 6,    # 3-0, 3-1, 3-2 W and their mirror losses
}


# ─────────────────────────────────────────────────────────────────────────────
# Parameter builders
# ─────────────────────────────────────────────────────────────────────────────

def make_AC_params(L: int, gamma: float = 1.0, eta: float = 0.0,
                   scale: float = 1.0, n_weight_categories: int = 1,
                   symmetric: bool = True) -> dict:
    """
    Build a JAX-compatible parameter dictionary for the AC model with L outcomes.

    Both cases fix the WLOG normalization  alpha_0=0,  delta_0=0,  delta_{L-1}=1.

    symmetric=True  (default)
    -------------------------
    Enforces home/away swap symmetry:
        alpha_y = alpha_{L-1-y}        (alpha symmetric)
        delta_y = 1 - delta_{L-1-y}   (delta anti-symmetric around 0.5)

    Free parameters:
        alpha : Na = (L-1)//2   (pairs of equal values; middle free if L odd)
        delta : Nd = (L-2)//2   (interior pairs; endpoints and middle fixed)

    For L=3 specifically: delta_full = [0, 0.5, 1] is fully determined (Nd=0).

    symmetric=False  ("free")
    -------------------------
    Only the WLOG normalization; no symmetry imposed.

    Free parameters:
        alpha : Na = L-1   (alpha_1 ... alpha_{L-1} free, alpha_0=0 fixed)
        delta : Nd = L-2   (delta_1 ... delta_{L-2} free, delta_0=0 and delta_{L-1}=1 fixed)
    """
    if symmetric:
        Na = (L - 1) // 2
        Nd = (L - 2) // 2

        Aalpha_rows = []
        for y in range(L):
            row = np.zeros(Na)
            mirror = L - 1 - y
            if y == 0 or y == L - 1:
                pass
            elif y < mirror:
                k = y - 1
                if k < Na:
                    row[k] = 1.0
            elif y > mirror:
                k = L - 2 - y
                if k < Na:
                    row[k] = 1.0
            else:
                k = y - 1
                if k < Na:
                    row[k] = 1.0
            Aalpha_rows.append(row)
        Aalpha = jnp.array(np.stack(Aalpha_rows))  # (L, Na)

        bdelta = jnp.arange(L, dtype=float) / (L - 1.0)
        Adelta_rows = []
        for y in range(L):
            row = np.zeros(Nd)
            mirror = L - 1 - y
            if y == 0 or y == L - 1:
                pass
            elif y == mirror:
                pass
            elif y < mirror:
                k = y - 1
                if k < Nd:
                    row[k] = +1.0
            else:
                k = L - 2 - y
                if k < Nd:
                    row[k] = -1.0
            Adelta_rows.append(row)
        Adelta = jnp.array(np.stack(Adelta_rows))  # (L, Nd)

        delta_init = np.zeros(Nd, dtype=float)
        alpha_init = np.zeros(Na, dtype=float)

    else:
        Na = L - 1
        Nd = L - 2

        Aalpha = jnp.concatenate([jnp.zeros((1, Na)),
                                  jnp.eye(Na)], axis=0)          # (L, Na)

        bdelta_arr        = np.zeros(L)
        bdelta_arr[L - 1] = 1.0
        bdelta = jnp.array(bdelta_arr)
        Adelta = jnp.concatenate([jnp.zeros((1, Nd)),
                                  jnp.eye(Nd),
                                  jnp.zeros((1, Nd))], axis=0)   # (L, Nd)

        delta_init = np.arange(1, L - 1, dtype=float) / (L - 1.0)
        alpha_init = np.zeros(Na, dtype=float)

    return {
        "scale":     scale,
        "gamma":     float(gamma),
        "eta":       float(eta),
        "Adelta":    Adelta,
        "bdelta":    bdelta,
        "delta":     delta_init,
        "Aalpha":    Aalpha,
        "alpha":     alpha_init,
        "weight":    np.ones(n_weight_categories, dtype=float),
        "L":         L,
    }


def make_CL_params(L: int, gamma: float = 1.0, eta: float = 0.0,
                   scale: float = 1.0, n_weight_categories: int = 1,
                   CDF: int = 0, symmetric: bool = True) -> dict:
    """
    Build a JAX-compatible parameter dictionary for the CL model with L outcomes.

    The CL model has Nc_tot = L-1 thresholds stored as c_full = Ac @ c.
    c_full must be STRICTLY INCREASING for valid probabilities.

    symmetric=True  (default)
    -------------------------
    Enforces home/away swap symmetry:  c_full[i] = -c_full[L-2-i]
    Free parameters: c : Nc = (L-1)//2 strictly positive upper-half thresholds.

    symmetric=False  ("free")
    -------------------------
    All L-1 thresholds free; Ac = identity.
    """
    Nc_tot = L - 1

    if symmetric:
        Nc = Nc_tot // 2
        Ac = np.zeros((Nc_tot, Nc), dtype=float)
        for i in range(Nc):
            Ac[i, Nc - 1 - i] = -1.0
            Ac[Nc_tot - Nc + i, i] = +1.0
        c_init = np.arange(1, Nc + 1, dtype=float) / (Nc + 1)
    else:
        Nc = Nc_tot
        Ac = np.eye(Nc_tot, dtype=float)
        c_init = np.arange(1, Nc + 1, dtype=float) / (Nc + 1) - 0.5

    return {
        "scale":  scale,
        "gamma":  float(gamma),
        "eta":    float(eta),
        "Ac":     jnp.array(Ac),
        "c":      jnp.array(c_init),
        "CDF":    CDF,
        "L":      L,
        "weight": np.ones(n_weight_categories, dtype=float),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data builder
# ─────────────────────────────────────────────────────────────────────────────

def make_regression_data(sport_type: str, year: int,
                         result_cat=None) -> tuple:
    """
    Load one season via manage_data and return (data, names, L).

    data = (X, y, u) where
        X : jnp array, shape M x T  (home +1, away -1; transposed from manage_data)
        y : jnp array, shape T      integer ordinal outcome in {0, ..., L-1}
        u : dict  {"hfa": ones(T), "category": zeros(T), "home_idx", "away_idx"}

    result_cat : None / "league" / "full" / int
        None or "league" uses the league-default result coding for the season.
    """
    if not _MANAGE_DATA_AVAILABLE:
        raise ImportError(
            "manage_data (SKF/manage_data.py) is not on sys.path.\n"
            "To fit an arbitrary matrix directly, use _fit_data(model, data, names, L, ...)\n"
            "where data = (X, y, u) is built by the caller."
        )
    if result_cat is None:
        result_cat = RESULT_CAT_DEFAULT.get(sport_type, "league")

    with contextlib.redirect_stdout(io.StringIO()):
        season = manage_data.get_season(sport_type, year)
        season.define_results(result_cat=result_cat)
        if isinstance(result_cat, int):
            L = result_cat
        else:
            L, _ = season.league_results(result_cat)
        season.regression_format()

    T = season.T
    X = jnp.array(season.X.T, dtype=float)    # manage_data: T x M  ->  M x T
    y = jnp.array(season.y,   dtype=int)

    X_np     = np.array(season.X)             # T x M
    home_idx = jnp.array(np.argmax(X_np, axis=1), dtype=int)
    away_idx = jnp.array(np.argmin(X_np, axis=1), dtype=int)

    u = {
        "hfa":      jnp.ones(T,  dtype=float),
        "category": jnp.zeros(T, dtype=int),
        "home_idx": home_idx,
        "away_idx": away_idx,
    }

    return (X, y, u), season.names, L


# ─────────────────────────────────────────────────────────────────────────────
# AC-specific helpers  (score-point vector xi'_y)
# ─────────────────────────────────────────────────────────────────────────────

def _cov_slices(free_vars: list, key_sizes) -> dict:
    """Reconstruct key → slice mapping from stored free_vars and key_sizes."""
    slices, start = {}, 0
    for k, s in zip(free_vars, map(int, key_sizes)):
        slices[k] = slice(start, start + s)
        start += s
    return slices


def _std_from_cov(cov: np.ndarray, slices: dict, key: str):
    """Std of one parameter block extracted from the full covariance matrix."""
    if key not in slices:
        return None
    sl = slices[key]
    return np.sqrt(np.maximum(np.diag(cov[sl, sl]), 0.0))


def _xi_AC_std_from_cov(L: int, symmetric: bool,
                        alpha_full, delta_full, eta, v_theta,
                        cov: np.ndarray, slices: dict):
    """Std of xi'_y for AC via the delta method.

    Differentiates xi' w.r.t. the expanded (full-L) alpha and delta, then
    chain-rules through Aalpha / Adelta (reconstructed from L and symmetric --
    no params_opt required).
    """
    if "alpha" not in slices and "delta" not in slices:
        return None

    pp_ref = make_AC_params(L, symmetric=symmetric)
    Aalpha = jnp.array(pp_ref["Aalpha"])   # (L, Na)
    Adelta = jnp.array(pp_ref["Adelta"])   # (L, Nd)
    Na, Nd = Aalpha.shape[1], Adelta.shape[1]

    mu_bar = 2.0 * float(np.sqrt(float(v_theta))) / float(np.sqrt(np.pi))
    eta_f  = float(eta)
    af     = jnp.array(np.asarray(alpha_full, dtype=float))
    df     = jnp.array(np.asarray(delta_full, dtype=float))

    def xi_from_expanded(a, d):
        xi = a + (eta_f + mu_bar) * d
        return (xi - xi[0]) / (xi[-1] - xi[0])

    Ja, Jd = jax.jacobian(xi_from_expanded, argnums=(0, 1))(af, df)
    J = jnp.concatenate([Ja @ Aalpha, Jd @ Adelta], axis=1)  # (L, Na+Nd)

    ad_idx = (list(range(slices["alpha"].start, slices["alpha"].stop))
              if "alpha" in slices else [])
    ad_idx += (list(range(slices["delta"].start, slices["delta"].stop))
               if "delta" in slices else [])
    cov_xi = J @ jnp.array(cov[np.ix_(ad_idx, ad_idx)]) @ J.T
    return np.array(jnp.sqrt(jnp.maximum(jnp.diag(cov_xi), 0.0)))


def _xi_CL_std_from_cov(L: int, symmetric: bool,
                        c_full: np.ndarray,
                        cov: np.ndarray, slices: dict):
    """Std of xi_y for CL via the delta method."""
    if "c" not in slices:
        return None

    pp_ref = make_CL_params(L, symmetric=symmetric)
    Ac     = jnp.array(pp_ref["Ac"])          # (L-1, Nc)
    cf     = jnp.array(np.asarray(c_full, dtype=float))

    def xi_from_c_full(cv):
        interior = (cv[1:] + cv[:-1] - cv[0]) / (2.0 * cv[-1])
        return jnp.concatenate([jnp.zeros(1), interior, jnp.ones(1)])

    J      = jax.jacobian(xi_from_c_full)(cf) @ Ac   # (L, Nc)
    sl     = slices["c"]
    cov_xi = J @ jnp.array(cov[sl, sl]) @ J.T
    return np.array(jnp.sqrt(jnp.maximum(jnp.diag(cov_xi), 0.0)))


def compute_xi_y(pp: dict, v_theta) -> np.ndarray:
    """Compute the normalised score-point vector xi'_y (AC model only).

    xi'_y = (alpha_y + (mu_bar + eta) * delta_y) / (alpha_{L-1} + mu_bar + eta)
    mu_bar = 2*sqrt(v_theta) / sqrt(pi)
    """
    alpha = np.array(pp["Aalpha"] @ pp["alpha"])
    delta = np.array(pp["Adelta"] @ pp["delta"] + pp["bdelta"])
    eta   = float(pp["eta"])

    mu_bar    = 2.0 * np.sqrt(float(v_theta)) / np.sqrt(np.pi)
    alpha_eff = alpha + eta * delta
    xi        = alpha_eff + mu_bar * delta
    xi_prime  = (xi - xi[0]) / (xi[-1] - xi[0])
    return xi_prime


def compute_xi_CL(c_full) -> np.ndarray:
    """Compute the score-point vector xi_y for the CL model.

    xi_0 = 0,  xi_{L-1} = 1,
    xi_y = (c_full[y] + c_full[y-1] - c_full[0]) / (2 * c_full[L-2])
           for y = 1, ..., L-2.

    (Eq. score.CL.neutral in the paper.)
    """
    c_full = np.asarray(c_full, dtype=float)
    L      = len(c_full) + 1          # number of outcomes
    xi     = np.zeros(L)
    xi[-1] = 1.0
    if L > 2:
        xi[1:-1] = (c_full[1:] + c_full[:-1] - c_full[0]) / (2.0 * c_full[-1])
    return xi


# Re-exported for backwards compatibility
predict_proba_AC = RR.predict_proba_AC


# ─────────────────────────────────────────────────────────────────────────────
# Inner fitting logic  (_fit_data)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_data(model: str, data: tuple, names: list, L: int,
              gamma_vec,
              symmetric:   bool       = True,
              opt_alpha:   bool       = True,
              opt_delta:   bool       = True,
              opt_eta:     bool|None  = None,
              method:      str        = "ml",
              init_params: list|None  = None,
              label:       str        = "",
              verbose:     bool       = True,
              reg_fn:      str        = "L2") -> dict:
    """Inner fitting logic for AC / CL / CLp models given pre-built (X, y, u) data.

    model : "AC"  -- Adjusting Categories (softmax)
            "CL"  -- Cumulative Logistic (proportional-odds / logit CDF)
            "CLp" -- Cumulative Probit (normal CDF)
    opt_alpha, opt_delta -- AC only; for CL/CLp the threshold vector c is always
        the free parameter (no separate alpha/delta distinction).
    opt_eta -- applies to AC, CL, and CLp; controls whether eta (home advantage)
        is included in the optimisation.

    Called by fit_season and fit_all_seasons_jointly. The caller is responsible
    for the verbose header line; this function prints only the post-fit summary.

    Returns a result dict with keys common to both models plus model-specific
    fields (alpha/delta/xi_y for AC; c for CL/CLp; None where not applicable).
    """
    X, y, u = data
    M, T = X.shape

    if opt_eta is None:
        # CL/CLp: eta optimised only for the symmetric variant (free CL can absorb
        #         HFA into the unconstrained threshold vector c, so eta is redundant)
        # AC:     eta only when symmetric (free AC can absorb HFA into alpha)
        opt_eta = symmetric

    if gamma_vec is None:
        gamma_vec = DEFAULT_GAMMA * (T / M)

    # ── Parameter construction ────────────────────────────────────────────
    if init_params is not None:
        params_list = [copy.deepcopy(pp) for pp in init_params]
    elif model == "AC":
        params_list = [make_AC_params(L, gamma=g, symmetric=symmetric)
                       for g in gamma_vec]
        if not symmetric:
            for pp in params_list:
                pp["eta"] = 0.0
    elif model in ("CL", "CLp"):
        params_list = [make_CL_params(L, gamma=g, symmetric=symmetric,
                                      CDF=_cl_cdf(model))
                       for g in gamma_vec]
        if not symmetric:
            for pp in params_list:
                pp["eta"] = 0.0
    else:
        raise ValueError(f"Unknown model {model!r}. Use 'AC', 'CL', or 'CLp'.")

    # ── Optimisation schedule ─────────────────────────────────────────────
    if model == "AC":
        Na         = len(params_list[0]["alpha"])
        Nd_delta   = len(params_list[0]["delta"])
        alpha_mask = np.ones(Na,       int)
        delta_mask = np.ones(Nd_delta, int)
        eta_mask   = np.ones(1,        int)

        block1 = {}
        if opt_alpha and Na > 0:
            block1["alpha"] = alpha_mask
        if opt_eta and symmetric:
            block1["eta"]   = eta_mask
        has_free       = opt_delta and Nd_delta > 0
        opt_scheduling = (([block1] if block1 else []) +
                          ([{"delta": delta_mask}] if has_free else []))
        em_block1      = {**block1, "gamma": np.ones(1, int)}
        em_scheduling  = ([em_block1] +
                          ([{"delta": delta_mask}] if has_free else []))

    else:  # CL / CLp
        Nc             = len(params_list[0]["c"])
        has_free       = True
        eta_mask       = np.ones(1, int)
        cl_block       = {"c": np.ones(Nc, int)}
        if opt_eta:
            cl_block["eta"] = eta_mask
        opt_scheduling = [cl_block]
        em_scheduling  = [{**cl_block, "gamma": np.ones(1, int)}]

    # ── Loss function and fitter namespace ────────────────────────────────
    ell_fn = RR.ac_ell if model == "AC" else RR.cl_ell
    fitter = RR.make_fitter(ell_fn, reg_fn)

    # ── Fit ───────────────────────────────────────────────────────────────
    if method == "em":
        pp_opt     = fitter.optimize_EM(
            copy.deepcopy(params_list[0]), data, em_scheduling)
        params_opt = [pp_opt]
        gamma_vec  = np.array([float(pp_opt["gamma"])])
    elif opt_scheduling:
        if method == "alo":
            params_opt = fitter.optimize_ALO(params_list, data, opt_scheduling)
        else:
            params_opt, _ = fitter.optimize_ML(params_list, data, opt_scheduling)
    else:
        params_opt = copy.deepcopy(params_list)

    # ── Evaluate ALO + extract parameters ────────────────────────────────
    n_gamma   = len(gamma_vec)
    alo       = {"full":    np.zeros(n_gamma),
                 "hfa":     np.zeros(n_gamma),
                 "neutral": np.zeros(n_gamma)}
    theta_all = np.zeros((M, n_gamma))
    eta_all   = np.zeros(n_gamma)
    hfa_mask  = np.array(u["hfa"]) == 1

    if model == "AC":
        alpha_all = np.zeros((L, n_gamma))
        delta_all = np.zeros((L, n_gamma))
    else:
        c_all = np.zeros((L - 1, n_gamma))   # full thresholds Ac @ c

    for k, pp in enumerate(params_opt):
        alo_val, alo_vec  = fitter.ALO(pp, data)
        alo["full"][k]    = float(alo_val)
        alo["hfa"][k]     = float(alo_vec[hfa_mask].mean())
        nc                = (~hfa_mask).sum()
        alo["neutral"][k] = float(alo_vec[~hfa_mask].mean()) if nc > 0 else 0.0
        theta_all[:, k]   = np.array(fitter.theta_hat(pp, data))
        eta_all[k]        = float(pp["eta"])
        if model == "AC":
            alpha_all[:, k] = np.array(pp["Aalpha"] @ pp["alpha"])
            delta_all[:, k] = np.array(pp["Adelta"] @ pp["delta"] + pp["bdelta"])
        else:
            c_all[:, k] = np.array(pp["Ac"] @ pp["c"])

    best_k         = int(np.argmin(alo["full"]))
    v_ns, v_sw     = fitter.v_theta_fun(params_opt[best_k], data)
    theta_best     = theta_all[:, best_k]
    v_theta_scalar = float(np.mean(theta_best**2) + np.mean(np.array(v_ns)))
    # xi_y kept for bootstrap use (AC only); display recomputes from saved params
    xi_y = (compute_xi_y(params_opt[best_k], v_theta_scalar)
            if model == "AC" else None)

    y_arr   = np.array(data[1], dtype=int)
    counts  = np.bincount(y_arr, minlength=L).astype(float)
    probs   = counts / counts.sum()
    entropy = -float(np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
    U_opt   = float(fitter.U_obj(params_opt[best_k], data))

    res = {
        "model":      model,
        "alo":        alo,
        "params_opt": params_opt,
        "gamma":      np.asarray(gamma_vec),
        "theta":      theta_all,
        "eta":        eta_all,
        "names":      names,
        "L":          L,
        "method":     method,
        "entropy":    entropy,
        "U_opt":      U_opt,
        "v_theta_ns": np.array(v_ns),
        "v_theta_sw": np.array(v_sw),
        "v_theta":    v_theta_scalar,
        # AC-specific (None for CL/CLp)
        "alpha":      alpha_all if model == "AC" else None,
        "delta":      delta_all if model == "AC" else None,
        "xi_y":       xi_y,       # used by bootstrap; display recomputes
        # CL/CLp-specific (None for AC)
        "c":          c_all if model in ("CL", "CLp") else None,
        # Marginal uncertainty (EM only; None otherwise) -- replaces all std_* fields
        "cov":        None,
        "free_vars":  None,
        "key_sizes":  None,
    }

    # ── Marginal uncertainty (EM only) ────────────────────────────────────
    if method == "em":
        pp_em     = params_opt[0]
        free_vars = list(dict.fromkeys(
            k for block in em_scheduling for k in block.keys()))
        _std_dict, cov, key_slices = fitter.marginal_uncertainty(
            pp_em, data, free_vars)
        key_sizes        = np.array([key_slices[k].stop - key_slices[k].start
                                     for k in free_vars])
        res["cov"]       = np.array(cov)
        res["free_vars"] = free_vars
        res["key_sizes"] = key_sizes

    if verbose and label:
        print_season_summary(res, label=label, symmetric=symmetric)

    return res


def _fit_AC_data(data: tuple, names: list, L: int,
                 gamma_vec,
                 symmetric:   bool       = True,
                 opt_alpha:   bool       = True,
                 opt_delta:   bool       = True,
                 opt_eta:     bool       = False,
                 method:      str        = "ml",
                 init_params: list|None  = None,
                 label:       str        = "",
                 verbose:     bool       = True,
                 reg_fn:      str        = "L2") -> dict:
    """Backward-compatible shim: delegates to _fit_data('AC', ...)."""
    return _fit_data("AC", data, names, L, gamma_vec,
                     symmetric=symmetric, opt_alpha=opt_alpha,
                     opt_delta=opt_delta, opt_eta=opt_eta,
                     method=method, init_params=init_params,
                     label=label, verbose=verbose, reg_fn=reg_fn)


# ─────────────────────────────────────────────────────────────────────────────
# Single-season fit
# ─────────────────────────────────────────────────────────────────────────────

def fit_season(sport_type: str, year: int,
               gamma_vec=None,
               result_cat=None,
               symmetric:  bool      = True,
               opt_alpha:  bool      = True,
               opt_delta:  bool      = True,
               opt_eta:    bool|None = None,
               method:     str       = "ml",
               model:      str       = "AC",
               save:       bool      = False,
               overwrite:  bool      = False,
               verbose:    bool      = True) -> dict:
    """Fit AC, CL, or CLp model for one season.

    model : "AC"  -- Adjusting Categories (default)
            "CL"  -- Cumulative Logistic (proportional-odds)
            "CLp" -- Cumulative Probit

    Results saved to:
        results/{sport_type}/{MODEL}_{sport_type}_{year}-{year+1}_{tag}.npz
    """
    model = model.upper()
    if opt_eta is None:
        # CL/CLp: eta optimised only for the symmetric variant (free CL can absorb
        #         HFA into the unconstrained threshold vector c, so eta is redundant)
        # AC:     eta only optimised for the symmetric variant (free AC has identifiability
        #         issues since alpha is unconstrained and can absorb any constant shift)
        opt_eta = symmetric
    out_dir = RESULTS_ROOT / sport_type
    tag     = _model_tag(symmetric, method)
    fname   = out_dir / f"{model}_{sport_type}_{year}-{year+1}_{tag}.npz"

    if save and fname.exists() and not overwrite:
        if verbose:
            print(f"  Loading {fname.name}")
        res = load_season(sport_type, year, symmetric=symmetric,
                          method=method, model=model)
        if verbose:
            print_season_summary(res, label=f"{sport_type} {year}-{year+1}",
                                 symmetric=symmetric)
        return res

    data, names, L = make_regression_data(sport_type, year, result_cat)
    X, y, u = data
    M, T = X.shape

    if verbose:
        model_str = "symmetric" if symmetric else "free"
        print(f"  {sport_type} {year}-{year+1}:  M={M}, T={T}, L={L}"
              f"  [{model}] [{model_str}]")

    res = _fit_data(model, data, names, L, gamma_vec,
                    symmetric=symmetric, opt_alpha=opt_alpha,
                    opt_delta=opt_delta, opt_eta=opt_eta,
                    method=method,
                    label=f"{sport_type} {year}-{year+1}",
                    verbose=verbose)

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_season_result(fname, res)
        if verbose:
            best_k = int(np.argmin(res["alo"]["full"]))
            print(f"  Saved: {fname.name}  "
                  f"(best gamma={res['gamma'][best_k]:.4f}, "
                  f"ALO={res['alo']['full'][best_k]:.4f})")

    return res


def _model_tag(symmetric: bool, method: str) -> str:
    """Filename tag: e.g. 'sym', 'free', 'sym_ml', 'free_em'."""
    base = "sym" if symmetric else "free"
    return base if method == "alo" else f"{base}_{method}"


def _save_season_result(fname, res) -> None:
    """Save a fit result dict to .npz (handles both AC and CL)."""
    model = res.get("model", "AC")
    kwargs = dict(
        model      = np.array(model),
        alo        = np.array(res["alo"],        dtype=object),
        params     = np.array(res["params_opt"], dtype=object),
        gamma      = res["gamma"],
        theta      = res["theta"],
        eta        = res["eta"],
        names      = np.array(res["names"]),
        L          = res["L"],
        v_theta_ns = res["v_theta_ns"],
        v_theta_sw = res["v_theta_sw"],
        v_theta    = res["v_theta"],
    )
    # AC-specific fields
    if res.get("alpha") is not None:
        kwargs["alpha"] = res["alpha"]
        kwargs["delta"] = res["delta"]
    # CL/CLp-specific fields
    if res.get("c") is not None:
        kwargs["c"] = res["c"]
    # Marginal uncertainty (EM only) -- single covariance matrix replaces all std_*
    if res.get("cov") is not None:
        kwargs["cov"]       = res["cov"]
        kwargs["free_vars"] = np.array(res["free_vars"])
        kwargs["key_sizes"] = np.array(res["key_sizes"])
    # Scalar diagnostics
    for key in ("entropy", "U_opt"):
        if res.get(key) is not None:
            kwargs[key] = res[key]
    if res.get("method") is not None:
        kwargs["method"] = np.array(res["method"])
    np.savez(fname, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-season loop
# ─────────────────────────────────────────────────────────────────────────────

def fit_all_seasons_separately(sport_type: str,
                    years,
                    gamma_vec=None,
                    result_cat=None,
                    symmetric: bool = True,
                    method:    str  = "ml",
                    model:     str  = "AC",
                    save: bool      = True,
                    overwrite: bool = False,
                    verbose: bool   = True) -> dict:
    """Fit AC, CL, or CLp model for every season in *years*.

    model : "AC" (default), "CL" (logistic), or "CLp" (probit)

    For each season:
      - If the .npz file exists and overwrite=False: load from disk and print summary.
      - Otherwise: fit, save, and print summary.

    Results saved to:
        results/{sport_type}/{MODEL}_{sport_type}_{year}-{year+1}_{tag}.npz
    """
    model   = model.upper()
    out_dir = RESULTS_ROOT / sport_type
    out_dir.mkdir(parents=True, exist_ok=True)

    model_tag   = _model_tag(symmetric, method)
    all_results = {}

    for year in years:
        fname = out_dir / f"{model}_{sport_type}_{year}-{year+1}_{model_tag}.npz"

        if fname.exists() and not overwrite:
            res = load_season(sport_type, year, symmetric=symmetric,
                              method=method, model=model, _fname=fname)
            if verbose:
                print_season_summary(res, label=f"{sport_type} {year}-{year+1}",
                                     symmetric=symmetric)
            all_results[year] = res
            continue

        if verbose:
            print(f"\nFitting {sport_type} {year}-{year+1} [{model}] ...")

        try:
            res = fit_season(sport_type, year,
                             gamma_vec=gamma_vec,
                             result_cat=result_cat,
                             symmetric=symmetric,
                             method=method,
                             model=model,
                             verbose=verbose)
        except Exception as exc:
            print(f"  ERROR {sport_type} {year}-{year+1}: {exc}")
            continue

        if save:
            _save_season_result(fname, res)
            if verbose:
                best_k = int(np.argmin(res["alo"]["full"]))
                print(f"  Saved: {fname.name}  "
                      f"(best gamma={res['gamma'][best_k]:.4f}, "
                      f"ALO={res['alo']['full'][best_k]:.4f})")

        all_results[year] = res
        jax.clear_caches()

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Jointly across all seasons (super-season)
# ─────────────────────────────────────────────────────────────────────────────

def _make_super_season(sport_type: str, years, result_cat=None) -> tuple:
    """Concatenate seasons into a super-season with block-diagonal structure.

    Each season's teams are tagged with the year (e.g. "ManCity_2005") so they
    are treated as independent across seasons.  The shared (alpha/delta/c, eta)
    are the only cross-season coupling in the subsequent fit.
    """
    y_list    = []
    hfa_list  = []
    home_list = []
    away_list = []
    names_all = []
    L_check   = None
    M_offset  = 0

    for year in years:
        (X_s, y_s, u_s), names_s, L_s = make_regression_data(
            sport_type, year, result_cat)
        M_s = X_s.shape[0]
        if L_check is None:
            L_check = L_s
        elif L_s != L_check:
            raise ValueError(
                f"Inconsistent L: year {year} has L={L_s}, expected {L_check}")
        y_list.append(np.array(y_s))
        hfa_list.append(np.array(u_s["hfa"]))
        home_list.append(np.array(u_s["home_idx"]) + M_offset)
        away_list.append(np.array(u_s["away_idx"]) + M_offset)
        names_all.extend([f"{n}_{year}" for n in names_s])
        M_offset += M_s

    M_super = M_offset
    y_super = jnp.array(np.concatenate(y_list), dtype=int)
    T_super = int(y_super.shape[0])

    X_super = jnp.zeros((M_super, T_super), dtype=jnp.int8)   # shape-only carrier

    u_super = {
        "hfa":      jnp.array(np.concatenate(hfa_list)),
        "category": jnp.zeros(T_super, dtype=int),
        "home_idx": jnp.array(np.concatenate(home_list), dtype=int),
        "away_idx": jnp.array(np.concatenate(away_list), dtype=int),
    }
    return (X_super, y_super, u_super), names_all, L_check


def fit_all_seasons_jointly(
    sport_type: str,
    years,
    name: str | None = None,
    gamma_vec=None,
    result_cat=None,
    symmetric: bool = True,
    method:    str  = "ml",
    model:     str  = "AC",
    save: bool      = True,
    overwrite: bool = False,
    n_boot: int     = 200,
    rng_seed: int   = 42,
    verbose: bool   = True,
    opt_eta: bool | None = None,
) -> dict:
    """Fit shared parameters jointly across all seasons via a super-season.

    model   : "AC" (default), "CL" (logistic), or "CLp" (probit)
    n_boot  : season-wise bootstrap replicates (0 = skip; not meaningful for EM)

    Results saved to:
        results/{sport_type}/{MODEL}_{name}_pooled_{tag}.npz
    """
    if method == "em" and n_boot > 0:
        raise ValueError(
            "n_boot > 0 is not meaningful with method='em': EM uncertainty "
            "is quantified via marginal_uncertainty (Hessian-based std). "
            "Set n_boot=0 for EM."
        )

    model     = model.upper()
    years     = list(years)
    name      = name or sport_type
    if opt_eta is None:
        opt_eta = symmetric
    model_tag = ("sym" if symmetric else "free") + ("" if method == "alo" else f"_{method}")
    if symmetric and not opt_eta:
        model_tag += "_eta0"          # venue-neutral: symmetric model, eta frozen at 0
    out_dir   = RESULTS_ROOT / sport_type
    out_dir.mkdir(parents=True, exist_ok=True)
    fname     = out_dir / f"{model}_{name}_pooled_{model_tag}.npz"

    # ── Point estimate ────────────────────────────────────────────────────
    res_fit = None
    if fname.exists() and not overwrite:
        res_fit = load_season(sport_type, 0, symmetric=symmetric,
                              method=method, model=model, _fname=fname)
        if verbose:
            print_season_summary(
                res_fit,
                label=f"{name} pooled ({len(years)} seasons)",
                symmetric=symmetric)

    if res_fit is None:
        if verbose:
            print(f"\nBuilding super-season: {name}  "
                  f"{len(years)} seasons [{model}] ...")
        data_super, names_super, L = _make_super_season(
            sport_type, years, result_cat)
        M_super, T_super = data_super[0].shape
        if verbose:
            model_str = "symmetric" if symmetric else "free"
            print(f"  M={M_super}, T={T_super}, L={L}  [{model_str}]")

        res_fit = _fit_data(
            model, data_super, names_super, L, gamma_vec,
            symmetric=symmetric, opt_eta=opt_eta,
            method=method,
            label=f"{name} pooled ({len(years)} seasons)",
            verbose=verbose)

        if save:
            _save_season_result(fname, res_fit)
            if verbose:
                best_k = int(np.argmin(res_fit["alo"]["full"]))
                print(f"  Saved: {fname.name}  "
                      f"(best gamma={res_fit['gamma'][best_k]:.4f}, "
                      f"ALO={res_fit['alo']['full'][best_k]:.4f})")

    # ── Season-wise bootstrap ─────────────────────────────────────────────
    boot = None
    if n_boot > 0:
        fname_boot = out_dir / f"{model}_{name}_pooled_{model_tag}_boot.npz"
        fname_ckpt = out_dir / f"{model}_{name}_pooled_{model_tag}_boot.partial.npz"

        if fname_boot.exists() and not overwrite:
            raw_b    = np.load(fname_boot)
            n_b      = raw_b["alo"].shape[0]
            alo_arr  = raw_b["alo"]
            alo_mean = float(raw_b["alo_mean"]) if "alo_mean" in raw_b else float(np.nanmean(alo_arr))
            alo_std  = float(raw_b["alo_std"])  if "alo_std"  in raw_b else float(np.nanstd(alo_arr, ddof=1))
            if model == "AC":
                boot = {
                    "alpha": raw_b["alpha"],
                    "delta": raw_b["delta"],
                    "eta":   raw_b["eta"],
                    "xi_y":  raw_b["xi_y"],
                    "alo":   alo_arr,
                    "mean": {
                        "alpha": raw_b["alpha_mean"],
                        "delta": raw_b["delta_mean"],
                        "eta":   float(raw_b["eta_mean"]),
                        "xi_y":  raw_b["xi_y_mean"],
                        "alo":   alo_mean,
                    },
                    "std": {
                        "alpha": raw_b["alpha_std"],
                        "delta": raw_b["delta_std"],
                        "eta":   float(raw_b["eta_std"]),
                        "xi_y":  raw_b["xi_y_std"],
                        "alo":   alo_std,
                    },
                }
            else:  # CL
                boot = {
                    "c":   raw_b["c"],
                    "eta": raw_b["eta"],
                    "alo": alo_arr,
                    "mean": {
                        "c":   raw_b["c_mean"],
                        "eta": float(raw_b["eta_mean"]),
                        "alo": alo_mean,
                    },
                    "std": {
                        "c":   raw_b["c_std"],
                        "eta": float(raw_b["eta_std"]),
                        "alo": alo_std,
                    },
                }
            if verbose:
                print(f"  Bootstrap loaded: {fname_boot.name}")
                print_season_summary(
                    res_fit, boot=boot,
                    label=f"{name} pooled ({len(years)} seasons)",
                    symmetric=symmetric)
        else:
            if verbose:
                print(f"\n  Bootstrap ({n_boot} season-wise replicates) ...")
            S          = len(years)
            L          = res_fit["L"]
            best_k     = int(np.argmin(res_fit["alo"]["full"]))
            gamma_boot = res_fit["gamma"][[best_k]]
            rng        = np.random.default_rng(rng_seed)

            all_years_b = [[years[i] for i in rng.integers(0, S, size=S)]
                           for _ in range(n_boot)]

            # Model-specific boot arrays
            if model == "AC":
                alpha_boot = np.full((L, n_boot), np.nan)
                delta_boot = np.full((L, n_boot), np.nan)
                xi_y_boot  = np.full((L, n_boot), np.nan)
            else:
                c_boot = np.full((L - 1, n_boot), np.nan)
            eta_boot = np.full(n_boot, np.nan)
            alo_boot = np.full(n_boot, np.nan)

            CKPT_EVERY = 10
            start_b    = 0

            if overwrite and fname_ckpt.exists():
                fname_ckpt.unlink()
            elif fname_ckpt.exists():
                raw_c   = np.load(fname_ckpt)
                start_b = int(raw_c["n_done"])
                if model == "AC":
                    alpha_boot[:, :start_b] = raw_c["alpha"][:, :start_b]
                    delta_boot[:, :start_b] = raw_c["delta"][:, :start_b]
                    xi_y_boot[:, :start_b]  = raw_c["xi_y"][:, :start_b]
                else:
                    c_boot[:, :start_b] = raw_c["c"][:, :start_b]
                eta_boot[:start_b] = raw_c["eta"][:start_b]
                alo_boot[:start_b] = raw_c["alo"][:start_b]
                if verbose:
                    print(f"  Resuming from checkpoint: {start_b}/{n_boot} reps done.",
                          flush=True)

            for b in range(start_b, n_boot):
                years_b = all_years_b[b]
                data_b = names_b = res_b = None
                try:
                    data_b, names_b, L_b = _make_super_season(
                        sport_type, years_b, result_cat)
                    warm = ([res_fit["params_opt"][best_k]]
                            if res_fit["params_opt"] is not None else None)
                    res_b = _fit_data(
                        model, data_b, names_b, L_b, gamma_boot,
                        symmetric=symmetric, opt_eta=symmetric,
                        method=method,
                        init_params=warm,
                        label="", verbose=False)
                    best_b = 0
                    if model == "AC":
                        alpha_boot[:, b] = res_b["alpha"][:, best_b]
                        delta_boot[:, b] = res_b["delta"][:, best_b]
                        xi_y_boot[:, b]  = res_b["xi_y"]
                    else:
                        c_boot[:, b] = res_b["c"][:, best_b]
                    eta_boot[b] = res_b["eta"][best_b]
                    alo_boot[b] = res_b["alo"]["full"][best_b]
                except Exception as exc:
                    if verbose:
                        print(f"  boot {b}: failed -- {exc}", flush=True)
                finally:
                    del data_b, names_b, res_b
                    gc.collect()
                    jax.clear_caches()

                if verbose:
                    print(f"  {b+1}/{n_boot} ...", end="\r", flush=True)

                if (b + 1) % CKPT_EVERY == 0:
                    if model == "AC":
                        np.savez(fname_ckpt,
                                 alpha=alpha_boot, delta=delta_boot,
                                 xi_y=xi_y_boot,
                                 eta=eta_boot, alo=alo_boot, n_done=b + 1)
                    else:
                        np.savez(fname_ckpt,
                                 c=c_boot,
                                 eta=eta_boot, alo=alo_boot, n_done=b + 1)
                    if verbose:
                        print(f"  checkpoint @ rep {b+1}/{n_boot}          ",
                              flush=True)

            if verbose:
                print()

            if model == "AC":
                boot = {
                    "alpha": alpha_boot,
                    "delta": delta_boot,
                    "eta":   eta_boot,
                    "xi_y":  xi_y_boot,
                    "alo":   alo_boot,
                    "mean": {
                        "alpha": np.nanmean(alpha_boot, axis=1),
                        "delta": np.nanmean(delta_boot, axis=1),
                        "eta":   float(np.nanmean(eta_boot)),
                        "xi_y":  np.nanmean(xi_y_boot,  axis=1),
                        "alo":   float(np.nanmean(alo_boot)),
                    },
                    "std": {
                        "alpha": np.nanstd(alpha_boot, axis=1, ddof=1),
                        "delta": np.nanstd(delta_boot, axis=1, ddof=1),
                        "eta":   float(np.nanstd(eta_boot, ddof=1)),
                        "xi_y":  np.nanstd(xi_y_boot,  axis=1, ddof=1),
                        "alo":   float(np.nanstd(alo_boot, ddof=1)),
                    },
                }
            else:  # CL
                boot = {
                    "c":   c_boot,
                    "eta": eta_boot,
                    "alo": alo_boot,
                    "mean": {
                        "c":   np.nanmean(c_boot, axis=1),
                        "eta": float(np.nanmean(eta_boot)),
                        "alo": float(np.nanmean(alo_boot)),
                    },
                    "std": {
                        "c":   np.nanstd(c_boot, axis=1, ddof=1),
                        "eta": float(np.nanstd(eta_boot, ddof=1)),
                        "alo": float(np.nanstd(alo_boot, ddof=1)),
                    },
                }

            if verbose:
                print_season_summary(
                    res_fit, boot=boot,
                    label=f"{name} pooled ({len(years)} seasons)",
                    symmetric=symmetric)

            if save:
                if model == "AC":
                    np.savez(fname_boot,
                             alpha      = boot["alpha"],
                             delta      = boot["delta"],
                             eta        = boot["eta"],
                             xi_y       = boot["xi_y"],
                             alo        = boot["alo"],
                             alpha_mean = boot["mean"]["alpha"],
                             delta_mean = boot["mean"]["delta"],
                             eta_mean   = np.array(boot["mean"]["eta"]),
                             xi_y_mean  = boot["mean"]["xi_y"],
                             alo_mean   = np.array(boot["mean"]["alo"]),
                             alpha_std  = boot["std"]["alpha"],
                             delta_std  = boot["std"]["delta"],
                             eta_std    = np.array(boot["std"]["eta"]),
                             xi_y_std   = boot["std"]["xi_y"],
                             alo_std    = np.array(boot["std"]["alo"]),
                             rng_seed   = np.array(rng_seed))
                else:  # CL
                    np.savez(fname_boot,
                             c        = boot["c"],
                             eta      = boot["eta"],
                             alo      = boot["alo"],
                             c_mean   = boot["mean"]["c"],
                             eta_mean = np.array(boot["mean"]["eta"]),
                             alo_mean = np.array(boot["mean"]["alo"]),
                             c_std    = boot["std"]["c"],
                             eta_std  = np.array(boot["std"]["eta"]),
                             alo_std  = np.array(boot["std"]["alo"]),
                             rng_seed = np.array(rng_seed))
                if verbose:
                    print(f"  Saved: {fname_boot.name}")
                if fname_ckpt.exists():
                    fname_ckpt.unlink()

    return {"fit": res_fit, "boot": boot}


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: load saved results
# ─────────────────────────────────────────────────────────────────────────────

def _safe_load_params(raw):
    """Load params_opt from npz, returning None if JAX version mismatch."""
    if "params" not in raw:
        return None
    try:
        return raw["params"].tolist()
    except Exception:
        return None


def load_season(sport_type: str, year: int, symmetric: bool = True,
                method: str = "alo", model: str = "AC",
                _fname=None):
    """Load a previously saved .npz result for AC or CL.

    model : "AC" (default) or "CL" -- selects the filename prefix.
    _fname: optional explicit path (used internally for pooled results).
    """
    if _fname is None:
        model  = model.upper()
        tag    = _model_tag(symmetric, method)
        _fname = RESULTS_ROOT / sport_type / f"{model}_{sport_type}_{year}-{year+1}_{tag}.npz"
    fname = pathlib.Path(_fname)
    if not fname.exists():
        raise FileNotFoundError(fname)
    raw = np.load(fname, allow_pickle=True)
    stored_model = str(raw["model"].item()) if "model" in raw else "AC"

    def _get(key):
        return raw[key] if key in raw else None

    return {
        "model":      stored_model,
        "alo":        raw["alo"].item(),
        "params_opt": _safe_load_params(raw),
        "gamma":      raw["gamma"],
        "theta":      raw["theta"],
        "names":      raw["names"].tolist(),
        "L":          int(raw["L"]),
        "v_theta_ns": _get("v_theta_ns"),
        "v_theta_sw": _get("v_theta_sw"),
        "v_theta":    float(_get("v_theta")) if _get("v_theta") is not None else None,
        # AC-specific
        "alpha":      _get("alpha"),
        "delta":      _get("delta"),
        # CL/CLp-specific
        "c":          _get("c"),
        # Marginal uncertainty (EM only)
        "cov":        _get("cov"),
        "free_vars":  list(_get("free_vars")) if _get("free_vars") is not None else None,
        "key_sizes":  _get("key_sizes"),
        # eta (home advantage; fixed at 0 for free model, optimised for symmetric)
        "eta":        _get("eta"),
        # Scalar diagnostics
        "entropy":    float(_get("entropy")) if _get("entropy") is not None else None,
        "U_opt":      float(_get("U_opt"))   if _get("U_opt")   is not None else None,
        "method":     str(_get("method").item()) if _get("method") is not None else None,
    }


def print_season_summary(res: dict, label: str = "",
                         symmetric: bool = True,
                         boot: dict | None = None) -> None:
    """Print fit summary for AC or CL (detected from res['model'])."""
    model   = res.get("model", "AC")
    best_k  = int(np.argmin(res["alo"]["full"]))
    gamma   = res["gamma"][best_k]
    alo     = res["alo"]["full"][best_k]
    eta     = res["eta"][best_k] if res.get("eta") is not None else float("nan")
    entropy = res.get("entropy")
    U_opt   = res.get("U_opt")
    method  = res.get("method", "ml")
    prefix  = f"  {label}  " if label else "  "

    theta_best     = res["theta"][:, best_k]
    v_theta_scalar = (float(np.mean(theta_best**2)) + float(np.mean(res["v_theta_ns"]))
                      if res.get("v_theta_ns") is not None else float("nan"))

    u_label     = "U*" if method == "em" else "U"
    info_parts  = [f"ALO={alo:.4f}"]
    if U_opt is not None:
        info_parts.append(f"{u_label}={U_opt:.4f}")
    if entropy is not None:
        info_parts.append(f"H={entropy:.4f}")
    print(f"{prefix}{',  '.join(info_parts)}")

    # ── Reconstruct covariance slices (both models) ──────────────────────────
    cov        = res.get("cov")
    _fk        = res.get("free_vars")
    _ks        = res.get("key_sizes")
    cov_slices = _cov_slices(_fk, _ks) if (cov is not None and _fk is not None) else {}

    if model == "AC":
        # ── AC display ────────────────────────────────────────────────────
        alpha_slice = slice(1, -1) if symmetric else slice(1, None)
        alpha = res["alpha"][:, best_k] if res.get("alpha") is not None else []
        delta = res["delta"][:, best_k] if res.get("delta") is not None else []

        # Recompute xi' from saved parameters (never use the cached disk value).
        if len(alpha) > 0 and not np.isnan(v_theta_scalar):
            mu_bar  = 2.0 * np.sqrt(v_theta_scalar) / np.sqrt(np.pi)
            xi_full = np.asarray(alpha) + (eta + mu_bar) * np.asarray(delta)
            xi_y    = (xi_full - xi_full[0]) / (xi_full[-1] - xi_full[0])
        else:
            xi_y = []

        _sym = {"alpha": "alpha", "delta": "delta", "eta": "eta", "gamma": "log gamma"}

        def _get_std_ac(k):
            if cov is not None:
                v = _std_from_cov(cov, cov_slices, k)
                if v is not None and v.size > 0:
                    return v
            if boot is None:
                return None
            if k == "alpha":
                return np.asarray(boot["std"]["alpha"])[alpha_slice]
            if k == "delta":
                return np.asarray(boot["std"]["delta"])[1:-1]
            if k == "eta":
                return np.atleast_1d(np.asarray(boot["std"]["eta"]))
            return None

        _param_vals = {
            "gamma": f"gamma={gamma:.4f}",
            "eta":   f"eta={eta:.3f}",
            "alpha": f"alpha={np.round(alpha, 3)}",
            "delta": f"delta={np.round(delta, 3)}",
        }
        fixed_parts = [val for k, val in _param_vals.items()
                       if _get_std_ac(k) is None or np.asarray(_get_std_ac(k)).size == 0]
        opt_parts   = [val for k, val in _param_vals.items()
                       if _get_std_ac(k) is not None and np.asarray(_get_std_ac(k)).size > 0]

        if fixed_parts:
            print(f"{prefix}  fixed:      {',  '.join(fixed_parts)}")
        if opt_parts:
            print(f"{prefix}  optimized:  {',  '.join(opt_parts)}")
        print(f"{prefix}  sqrt(v_theta)={np.sqrt(v_theta_scalar):.4f}"
              f"  ->  xi'={np.round(xi_y, 3)}")

        std_parts = []
        for k, sym in _sym.items():
            v = _get_std_ac(k)
            if v is not None and np.asarray(v).size > 0:
                std_parts.append(f"std({sym})={np.round(np.asarray(v), 4)}")
        if std_parts:
            print(f"{prefix}  std:        {',  '.join(std_parts)}")

        # xi' uncertainty: Hessian-based (EM) or bootstrap
        std_xi = None
        if cov is not None and len(alpha) > 0:
            std_xi = _xi_AC_std_from_cov(
                res["L"], symmetric, alpha, delta, eta, v_theta_scalar,
                cov, cov_slices)
        if std_xi is None and boot is not None:
            std_xi = boot["std"].get("xi_y")
        if std_xi is not None and len(xi_y) > 0:
            xi_interior  = np.array(xi_y)[1:-1]
            std_interior = np.array(std_xi)[1:-1]
            xi_str = "[" + ",  ".join(
                f"{x:.3f}+-{s:.3f}" for x, s in zip(xi_interior, std_interior)
            ) + "]"
            print(f"{prefix}  xi' (+-std) = {xi_str}")

    else:
        # ── CL / CLp display ──────────────────────────────────────────────
        c_full = res["c"][:, best_k] if res.get("c") is not None else []

        def _get_std_cl(k):
            if cov is not None:
                v = _std_from_cov(cov, cov_slices, k)
                if v is not None and v.size > 0:
                    return v
            if boot is None:
                return None
            if k == "c" and "std" in boot and "c" in boot["std"]:
                return np.asarray(boot["std"]["c"])
            if k == "eta" and "std" in boot and "eta" in boot["std"]:
                return np.atleast_1d(np.asarray(boot["std"]["eta"]))
            return None

        _param_vals_cl = {
            "gamma": f"gamma={gamma:.4f}",
            "eta":   f"eta={eta:.3f}",
            "c":     f"c_full={np.round(c_full, 3)}",
        }
        fixed_parts_cl = [val for k, val in _param_vals_cl.items()
                          if _get_std_cl(k) is None or np.asarray(_get_std_cl(k)).size == 0]
        opt_parts_cl   = [val for k, val in _param_vals_cl.items()
                          if _get_std_cl(k) is not None and np.asarray(_get_std_cl(k)).size > 0]
        if fixed_parts_cl:
            print(f"{prefix}  fixed:      {',  '.join(fixed_parts_cl)}")
        if opt_parts_cl:
            print(f"{prefix}  optimized:  {',  '.join(opt_parts_cl)}")

        # Recompute xi from c_full (never cached on disk).
        xi_cl  = compute_xi_CL(c_full) if len(c_full) > 0 else []
        sqrt_v = np.sqrt(v_theta_scalar)
        if len(xi_cl) > 0:
            print(f"{prefix}  sqrt(v_theta)={sqrt_v:.4f}"
                  f"  ->  xi={np.round(xi_cl, 3)}")
        else:
            print(f"{prefix}  sqrt(v_theta)={sqrt_v:.4f}")

        _cl_sym = {"c": "c", "gamma": "log gamma", "eta": "eta"}
        std_parts_cl = []
        for k, sym in _cl_sym.items():
            v = _get_std_cl(k)
            if v is not None and np.asarray(v).size > 0:
                std_parts_cl.append(f"std({sym})={np.round(np.asarray(v), 4)}")
        if std_parts_cl:
            print(f"{prefix}  std:        {',  '.join(std_parts_cl)}")

        # xi uncertainty: Hessian-based (EM) or bootstrap
        std_xi_cl = None
        if cov is not None and len(c_full) > 0:
            std_xi_cl = _xi_CL_std_from_cov(
                res["L"], symmetric, c_full, cov, cov_slices)
        if std_xi_cl is None and boot is not None:
            std_xi_cl = boot["std"].get("xi_y")
        if std_xi_cl is not None and len(xi_cl) > 0:
            xi_interior  = np.array(xi_cl)[1:-1]
            std_interior = np.array(std_xi_cl)[1:-1]
            if len(xi_interior) > 0:
                xi_str = "[" + ",  ".join(
                    f"{x:.3f}+-{s:.3f}" for x, s in zip(xi_interior, std_interior)
                ) + "]"
                print(f"{prefix}  xi (+-std) = {xi_str}")


def best_theta(sport_type: str, year: int, symmetric: bool = True,
               model: str = "AC"):
    """Return the team-strength vector for the gamma with lowest ALO loss."""
    res    = load_season(sport_type, year, symmetric=symmetric, model=model)
    best_k = int(np.argmin(res["alo"]["full"]))
    return res["theta"][:, best_k], res["names"], res["gamma"][best_k]


# ─────────────────────────────────────────────────────────────────────────────
# Verification utilities  (AC only)
# ─────────────────────────────────────────────────────────────────────────────

def scan_evidence(delta_grid,
                        name:           str   = "",
                        data                  = None,
                        gamma_init:     float = 1.0,
                        alpha_init            = None,
                        n_em_iter:      int   = 30,
                        out_path              = None,
                        verbose:        bool  = True):
    """Verification utility: scan delta on a grid and recover p(delta|y) for the AC model.

    For each delta on delta_grid, alpha and gamma are jointly optimised by
    marginal ML (EM).  The delta-posterior is:
        p(delta|y) proportional to exp(-U_obj(delta))

    See Optimize_AC_models.py / optimize_models.py docstring for full details.
    """
    delta_grid = np.asarray(delta_grid)
    n_delta    = len(delta_grid)

    if data is None:
        if not name:
            raise ValueError("Either data or name must be provided.")
        sport_type = _base_sport(name)
        data, _names, L = _make_super_season(
            sport_type, list(SEASON_CONFIG_JOINT[name]))
    else:
        L = int(data[1].max()) + 1
        if alpha_init is not None:
            L_from_alpha = len(alpha_init) + 1
            if L_from_alpha != L:
                raise ValueError(
                    f"alpha_init length implies L={L_from_alpha} "
                    f"but data implies L={L}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Evidence delta-scan: {name or '(data provided)'}  (L={L})")
        print(f"  delta grid  : {np.round(delta_grid, 3)}")
        print(f"  gamma_init  : {gamma_init:.4f}")
        print(f"{'='*60}", flush=True)

    t0 = time.time()

    delta_evidence = np.full(n_delta, np.nan)
    alpha_at_delta = []
    gamma_at_delta = np.full(n_delta, np.nan)

    Na = L - 1
    Nd = L - 2
    opt_scheduling = [{"alpha": np.ones(Na, int), "gamma": np.ones(1, int)}]

    alpha_warm = alpha_init
    gamma_warm = gamma_init

    for d_idx, delta1 in enumerate(delta_grid):
        pp = make_AC_params(L, gamma=gamma_warm, symmetric=False)
        delta_free    = np.zeros(Nd)
        delta_free[0] = float(delta1)
        pp["delta"]   = jnp.array(delta_free)
        if alpha_warm is not None:
            pp["alpha"] = jnp.array(alpha_warm)

        pp_opt = RR.optimize_EM(pp, data, opt_scheduling, n_em_iter=n_em_iter)

        ev = RR.U_obj(pp_opt, data)
        delta_evidence[d_idx] = ev
        alpha_at_delta.append(np.array(pp_opt["alpha"]))
        gamma_at_delta[d_idx] = float(pp_opt["gamma"])

        alpha_warm = np.array(pp_opt["alpha"])
        gamma_warm = float(pp_opt["gamma"])

        if verbose:
            print(f"  delta={delta1:.3f}  ev={ev:.5f}  "
                  f"gamma*={gamma_warm:.4f}  ({time.time()-t0:.0f}s)", flush=True)

        if (d_idx + 1) % 5 == 0:
            jax.clear_caches()

    alpha_at_delta = np.stack(alpha_at_delta, axis=0)

    log_post  = -delta_evidence
    log_post -= log_post.max()
    post      = np.exp(log_post)
    post     /= post.sum()

    delta_star = delta_grid[int(np.argmax(post))]

    if verbose:
        print(f"\n  delta*   = {delta_star:.4f}  (MAP)")
        print(f"  p(delta|y) = {np.round(post, 3)}")
        print(f"  Total wall time: {time.time()-t0:.0f}s", flush=True)

    result = dict(
        delta_grid     = delta_grid,
        delta_evidence = delta_evidence,
        alpha_at_delta = alpha_at_delta,
        gamma_at_delta = gamma_at_delta,
        log_post       = log_post,
        post           = post,
        delta_star     = float(delta_star),
        name           = name,
    )

    if out_path is not None:
        np.savez(out_path, **result)
        if verbose:
            print(f"  Saved: {out_path}", flush=True)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    do_DEBUG = os.environ.get("SLURM_JOB_ID") is None
    if do_DEBUG:
        if False:   ### separate
            sys.argv = ["optimize_models.py",
                        "--mode", "separate",
                        "--name", "Division1",
                        "--model", "AC",
                        "--method", "em",
                        "--n-boot", "0"]
        if True:   ### joint
            sys.argv = ["optimize_models.py",
                        "--mode", "joint",
                        "--name", "SuperLega",   #"Bundesliga_2pt", "NHL_4"
                        # "--overwrite",
                        "--symmetric",
                        "--model", "CL",
                        "--method", "em", "--n-boot", "0"
                        ]

    ap = argparse.ArgumentParser(
        description="Fit AC, CL, or CLp ordinal regression models (joint or per-season).")
    ap.add_argument("--mode",      default="joint",
                    choices=["joint", "separate"],
                    help="Fit mode: 'joint' (super-season) or 'separate' (per-season).")
    ap.add_argument("--model",     default="AC", choices=["AC", "CL", "CLp"],
                    help="Model: 'AC' (Adjusting Categories), 'CL' (Cumulative Logistic), "
                         "or 'CLp' (Cumulative Probit).")
    ap.add_argument("--method",    default="ml",    choices=["alo", "ml", "em"],
                    help="Optimisation method: 'ml' (BFGS), 'alo' (ALO Newton), "
                         "or 'em' (marginal ML via EM).")
    ap.add_argument("--symmetric", action="store_true", default=False,
                    help="Enforce home/away symmetry constraints.")
    ap.add_argument("--name",      default=None,
                    help="Single SEASON_CONFIG[_JOINT] key to run (default: all keys).")
    ap.add_argument("--n-boot",    type=int, default=200,
                    help="Bootstrap replicates per season (0 = skip bootstrap).")
    ap.add_argument("--overwrite", action="store_true", default=False,
                    help="Overwrite existing result files.")
    ap.add_argument("--eta0", action="store_true", default=False,
                    help="Freeze eta=0 (venue-neutral); requires --symmetric and --mode joint.")
    args = ap.parse_args()
    if args.eta0 and not args.symmetric:
        ap.error("--eta0 requires --symmetric (venue-neutral = symmetric model, eta frozen at 0).")
    if args.eta0 and args.mode != "joint":
        ap.error("--eta0 is only supported in --mode joint.")

    gamma_vec   = np.logspace(-2, 1, 15)
    MODE        = args.mode
    METHOD      = args.method
    MODEL       = args.model.upper()
    symmetric   = args.symmetric
    n_boot      = args.n_boot
    overwrite   = args.overwrite

    model_label = "symmetric" if symmetric else "free"
    if MODE == "separate":
        if args.name is not None:
            if args.name not in SEASON_CONFIG:
                raise ValueError(
                    f"Unknown --name {args.name!r}. "
                    f"Available keys: {list(SEASON_CONFIG)}")
            sep_items = [(args.name, SEASON_CONFIG[args.name])]
        else:
            sep_items = list(SEASON_CONFIG.items())
        for sport, years in sep_items:
            print(f"\n{'='*60}")
            print(f"  {sport}  [{MODEL}] [{model_label}]  [{MODE}]  [{METHOD}]")
            print(f"{'='*60}")
            fit_all_seasons_separately(sport, years,
                                       gamma_vec=gamma_vec,
                                       symmetric=symmetric,
                                       method=METHOD,
                                       model=MODEL,
                                       save=True,
                                       overwrite=overwrite,
                                       verbose=True)
    elif MODE == "joint":
        if args.name is not None:
            if args.name not in SEASON_CONFIG_JOINT:
                raise ValueError(
                    f"Unknown --name {args.name!r}. "
                    f"Available keys: {list(SEASON_CONFIG_JOINT)}")
            keys = [args.name]
        else:
            keys = list(SEASON_CONFIG_JOINT.keys())

        for name in keys:
            years      = SEASON_CONFIG_JOINT[name]
            sport      = _base_sport(name)
            result_cat = RESULT_CAT_JOINT.get(name, None)
            print(f"\n{'='*60}")
            print(f"  {name}  [{MODEL}] [{model_label}]  [{MODE}]  [{METHOD}]")
            print(f"{'='*60}")
            fit_all_seasons_jointly(sport, years,
                                    name=name,
                                    gamma_vec=gamma_vec,
                                    result_cat=result_cat,
                                    symmetric=symmetric,
                                    method=METHOD,
                                    model=MODEL,
                                    save=True,
                                    overwrite=overwrite,
                                    n_boot=n_boot,
                                    opt_eta=(False if args.eta0 else None),
                                    verbose=True)
