"""
RegularizedRegression.py
========================
Core fitting engine for ordinal regression models (AC, CL, FIVB).

Model dispatch
--------------
All JIT-compiled functions accept `ell_fn` as their first **static** argument
(declared via ``static_argnums=(0,)``).  Module-level aliases bind ``ell_fn=ac_ell``
so all existing call-sites continue to work unchanged:

    theta_hat(pp, data)         # AC model — call unchanged
    J_obj(theta, data, pp)      # AC model — call unchanged
    optimize_ML(pp, data, sch)  # AC model — call unchanged

To use a different model (e.g. CL), supply ``ell_fn`` explicitly:

    _theta_hat(cl_ell, pp, data)
    optimize_EM(pp, data, sch, ell_fn=cl_ell)

Or use ``make_fitter(ell_fn="AC", reg_fn="L2")`` to get a namespace of pre-bound functions.

Elementary loss functions
-------------------------
    ac_ell(z, y, params) -> scalar   AC (softmax) log-loss for one game
    cl_ell(z, y, params) -> scalar   CL (cumulative logit/probit) log-loss

Each has signature ``(z, y, params) -> scalar`` where z is the skill difference
*after* scale and HFA have been applied by the caller (_loss_fun).
"""

import copy

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import scipy
import scipy.optimize
from functools import partial
from jax import grad, hessian, jit, vmap, config
from types import SimpleNamespace

# FIVB figures module is only needed by SG_ranking (legacy) — guard it.
try:
    import FIVB_paper_figures as _FIVB
    _FIVB_AVAILABLE = True
except ImportError:
    _FIVB = None
    _FIVB_AVAILABLE = False

do_JIT = True
if not do_JIT:
    print("JIT is disabled")
    config.update("jax_disable_jit", True)


# ── Gauss-Hermite quadrature constants ───────────────────────────────────────
_GH_N = 10
_GH_NODES_NP, _GH_WEIGHTS_NP = np.polynomial.hermite.hermgauss(_GH_N)
_GH_NODES   = jnp.array(_GH_NODES_NP)    # (10,)
_GH_WEIGHTS = jnp.array(_GH_WEIGHTS_NP)  # (10,)  sum = √π

# ── Gauss-Laguerre quadrature constants ──────────────────────────────────────
# Approximates  ∫₀^∞ exp(−u) f(u) du ≈ Σᵢ wᵢ f(uᵢ)
_GL_N = 20
_GL_NODES_NP, _GL_WEIGHTS_NP = np.polynomial.laguerre.laggauss(_GL_N)
_GL_NODES   = jnp.array(_GL_NODES_NP)    # (20,)  positive nodes
_GL_WEIGHTS = jnp.array(_GL_WEIGHTS_NP)  # (20,)  positive weights


# ═══════════════════════════════════════════════════════════════════════════════
# Elementary loss functions
# ─────────────────────────────────────────────────────────────────────────────
# Signature:  ell_fn(z, y, params) -> scalar
#   z      : scalar skill difference (scale + HFA already applied by _loss_fun)
#   y      : integer outcome in {0, …, L-1}
#   params : full parameter dict
#
# Companion probability functions:
#   ac_prob(z, params) -> (L,)   probability vector (no y needed)
# ═══════════════════════════════════════════════════════════════════════════════

@jit
def _ac_logits(z, params):
    """AC model logit vector v = alpha_full + delta_full * z.

    Shared by ac_ell (loss) and ac_prob (probabilities) so the parameter
    expansion is defined exactly once.
    """
    alpha_full = params["Aalpha"] @ params["alpha"]
    delta_full = params["Adelta"] @ params["delta"] + params["bdelta"]
    return alpha_full + delta_full * z                    # (L,)


@jit
def ac_ell(z, y, params):
    """AC model: negative log-loss for one game.  -log P(Y=y | z)."""
    v = _ac_logits(z, params)
    return jax.scipy.special.logsumexp(v) - v[y]


@jit
def ac_prob(z, params):
    """AC model: probability vector P(Y=y | z) for one game.  Returns (L,)."""
    return jax.nn.softmax(_ac_logits(z, params))


@jit
def cl_ell(z, y, params):
    """CL model (cumulative logit/probit): negative log-loss for one game.

    Convention (matches AC model):  y=0 = away win (low z),  y=L-1 = home win (high z).
    Thresholds c_full = Ac @ c must be strictly increasing.

    NOTE — FIVB outcome convention is REVERSED (y=0 = home win).  Pass y→(L-1-y)
    when using cl_ell on FIVB data, or use FIVB_calculation.Qy_CL directly.

    params must contain:
        "Ac"  : (L-1, Nc) threshold expansion matrix
        "c"   : (Nc,) free threshold parameters; full thresholds = Ac @ c
        "CDF" : 0 = normal (probit), 1 = logistic

    Direct evaluation: ell_log = −log P(Y=· | z) at the SIGNED argument cc − z.
    Correct for arbitrary (a)symmetric thresholds — no symmetry fold.  (The old
    fold onto |z| was only valid for antisymmetric c and silently symmetrised
    the free model; see _ell_CL_log.)
    """
    cc      = params["Ac"] @ params["c"]
    ell_log = _ell_CL_log(cc - z, params)
    return ell_log[y]


@jit
def cl_prob(z, params):
    """CL model: probability vector P(Y=y | z) for one game.  Returns (L,).

    Direct evaluation at the signed argument cc − z; valid for any thresholds.
    """
    cc = params["Ac"] @ params["c"]
    return jnp.exp(-_ell_CL_log(cc - z, params))


def _log1mexp(s):
    """Stable log(1 − exp(−s)) for s > 0  (Mächler 2012, split at log 2)."""
    return jnp.where(s > jnp.log(2.0),
                     jnp.log1p(-jnp.exp(-s)),
                     jnp.log(-jnp.expm1(-s)))


@jit
def _ell_CL_log(zz, params):
    """−log P(Y=k | z) for every outcome k = 0 … L-1 under the CL model.

    Argument  zz = cc − z  (thresholds minus latent), length L-1, increasing.
    With  F_k = F(zz_k):
        P(Y=0)   = F_0
        P(Y=k)   = F_k − F_{k-1}      (1 ≤ k ≤ L-2)
        P(Y=L-1) = 1 − F_{L-2}
    Stable throughout: log F_k via the log-CDF, the top bin 1−F_{L-2} via the
    survival function (no cancellation), interior differences via log1mexp.
    """
    logF = jnp.where(params["CDF"] == 0,
                     jax.scipy.stats.norm.logcdf(zz),
                     jax.nn.log_sigmoid(zz))               # log F_k
    logS = jnp.where(params["CDF"] == 0,
                     jax.scipy.stats.norm.logcdf(-zz),
                     jax.nn.log_sigmoid(-zz))              # log(1 − F_k)
    mid  = logF[1:] + _log1mexp(logF[1:] - logF[:-1])      # log P(Y=k), k=1..L-2
    logP = jnp.concatenate([logF[:1], mid, logS[-1:]])     # length L
    return -logP


@jit
def fivb_loss(z, y, params):
    """FIVB scoring-rule loss (legacy)."""
    rr = params["Ar"] @ params["r"]
    cc = params["Ac"] @ params["c"]
    dr = -jnp.diff(rr)
    zz = z + cc
    psi = jax.scipy.stats.norm.cdf(zz) * zz + jax.scipy.stats.norm.pdf(zz)
    return jnp.dot(psi, dr) + jnp.dot(rr[-1] - rr[y], z)


# Backwards-compatible aliases for code that imports by old names
logarithmic_loss_AC = ac_ell
logarithmic_loss_CL = cl_ell


@partial(jit, static_argnums=(0, 1))
def _prob_from_ell(ell_fn, L, z, params):
    """Generic (L,) probability vector derived from ell_fn.

    P(Y=y | z) = exp(-ell_fn(z, y, params))  for y = 0 … L-1.

    L must be a static Python int (passed explicitly so it is never traced).
    XLA eliminates repeated shared sub-expressions (e.g. logsumexp for AC)
    so the compiled cost is equivalent to a hand-written prob function.
    """
    return jnp.exp(-vmap(lambda y: ell_fn(z, y, params))(jnp.arange(L)))


def _predict_proba(prob_fn, pp, z, hfa=None):
    """Vectorised probability matrix using an explicit prob_fn(z, params) -> (L,).

    Standalone utility; used by predict_proba_AC / predict_proba_CL which
    pass their hand-written ac_prob / cl_prob.
    """
    z_arr   = jnp.atleast_1d(jnp.asarray(z, dtype=float))
    hfa_arr = jnp.ones_like(z_arr) if hfa is None else \
              jnp.broadcast_to(jnp.asarray(hfa, dtype=float), z_arr.shape)
    z_eff = z_arr / pp["scale"] + pp["eta"] * hfa_arr          # (T,)
    return vmap(prob_fn, in_axes=[0, None])(z_eff, pp)         # (T, L)


def _predict_proba_generic(ell_fn, pp, z, hfa=None):
    """Generic predict_proba derived from ell_fn alone.

    Extracts pp["L"] (must be a Python int) before vmap traces pp,
    then delegates to _prob_from_ell.  Used by make_fitter so the
    namespace is the single model handle — no separate prob_fn needed.
    """
    L = pp["L"]                                                # concrete int, extracted here
    _prob = partial(_prob_from_ell, ell_fn, L)                 # L bound before vmap traces pp
    return _predict_proba(_prob, pp, z, hfa)


# Standalone model-specific aliases (use when calling directly, not via namespace)
predict_proba_AC = partial(_predict_proba, ac_prob)
predict_proba_CL = partial(_predict_proba, cl_prob)


# ═══════════════════════════════════════════════════════════════════════════════
# Regularisation
# ═══════════════════════════════════════════════════════════════════════════════

@jit
def regularization_fun(theta, params):
    """L2 (ridge) penalty: (γ/2)||θ||².

    Corresponds to a Gaussian prior N(0, 1/γ) on each skill.  The marginal ML
    update for γ is closed-form: γ = M / (Σ_m θ̂_m² + Σ_m v_m).
    """
    return 0.5 * params["gamma"] * jnp.sum(theta ** 2)


@jit
def regularization_fun_vg(theta, params):
    """Virtual-games (fictitious win + loss) penalty.

    Each player is augmented with γ fictitious wins and γ fictitious losses
    against a neutral opponent (θ_n = 0) at a neutral venue (η = 0).  The
    negative log-prior is

        -log f(θ) = -γ Σ_m [log P_{L-1}(z_m) + log P_0(z_m)],  z_m = θ_m / scale.

    Its gradient w.r.t. θ_m is  -γ (1 - 2 G(z_m)) / scale, which vanishes at
    θ_m = 0 (since G(0) = 1/2) and pulls every player toward the prior mean.
    For L = 2 this reduces to the logistic (beta-prior) regularisation that
    underlies the Colley method.  For L > 2 it involves G, which is not a
    logistic, so the prior is "logistic-style but not exactly logistic."

    Note: the marginal ML update for γ is not closed-form with this prior
    (the log-normaliser has no analytic expression for L > 2).  Set γ by
    cross-validation or treat it as a fixed hyperparameter.
    """
    gamma = params["gamma"]
    scale = params["scale"]
    z_m   = theta / scale                                    # (M,)
    def _ell_win_loss(z):
        v     = _ac_logits(z, params)
        log_Z = jax.scipy.special.logsumexp(v)
        return (log_Z - v[-1]) + (log_Z - v[0])             # ell_win + ell_loss
    return gamma * jnp.sum(vmap(_ell_win_loss)(z_m))


# ── Log-normalizer functions ──────────────────────────────────────────────────
# Each returns the value to add to J_val + 0.5*log_det_H in U_obj, i.e.,
# the negative log normalizer of the prior (up to constants independent of γ).

def log_Z_ridge(pp, M):
    """Log normalizer for the ridge prior: -M/2 · log γ.

    Returns a JAX scalar so that jax.grad / jax.hessian can differentiate
    through it when called from inside U_frozen (marginal_uncertainty).
    """
    return -0.5 * M * jnp.log(jnp.asarray(pp["gamma"]))


def log_Z_binary(pp, M):
    """Exact log normalizer for the virtual-games prior when L=2, alpha=0.

    For binary outcomes with delta=[0,1], the per-player prior is
        p̃(θ) = σ(θ/s)^γ · σ(−θ/s)^γ,
    whose integral over θ has the closed form (Beta/Gamma identity):
        Z_per = s · 2^{1−2γ} · √π · Γ(γ) / Γ(γ+½)

    This is exact for all γ>0 and avoids the factor-2 error in log(γ) that the
    Laplace approximation produces for heavy-tailed (Laplace-like) priors.
    Fully JAX-traceable.
    """
    gamma   = pp["gamma"]
    scale   = pp["scale"]
    log_Z_per = (
        jnp.log(scale)
        + (1.0 - 2.0 * gamma) * jnp.log(2.0)
        + 0.5 * jnp.log(jnp.pi)
        + jax.scipy.special.gammaln(gamma)
        - jax.scipy.special.gammaln(gamma + 0.5)
    )
    return M * log_Z_per


def d_log_Z_binary_d_gamma(pp, M):
    """Derivative of the binary VG log-normalizer w.r.t. γ.

    Starting from
        log Z_per = log s + (1−2γ)·log 2 + ½·log π + lgamma(γ) − lgamma(γ+½)

    differentiating term by term:
        d/dγ [(1−2γ)·log 2]    =  −2·log 2
        d/dγ [lgamma(γ)]       =   ψ(γ)          (digamma)
        d/dγ [−lgamma(γ+½)]    =  −ψ(γ+½)

    so
        d log Z_per / dγ = −2·log 2 + ψ(γ) − ψ(γ+½)

    and for M independent players:
        d log Z / dγ = M · (−2·log 2 + ψ(γ) − ψ(γ+½))

    Note: since Z = ∫ exp(−γ·ell_vg) dθ, we have
        d log Z / dγ = −E_prior[ell_vg]
    so this also gives the expected VG loss under the prior (with a sign flip).
    """
    gamma = pp["gamma"]
    d_log_Z_per = (
        -2.0 * jnp.log(2.0)
        + jax.scipy.special.digamma(gamma)
        - jax.scipy.special.digamma(gamma + 0.5)
    )
    return M * d_log_Z_per


def E_prior_ell_vg_binary(pp, M):
    """Expected virtual-game loss under the binary VG prior.

    By the log-partition identity  E_prior[ell_vg] = −d log Z / dγ:

        E_prior[ell_vg(θ/s)] = 2·log 2 − ψ(γ) + ψ(γ+½)

    Limiting behaviour:
        γ → 0 :  ψ(γ) ≈ −1/γ   →   E ≈ 1/γ          (diverges; prior is flat)
        γ → ∞ :  ψ(γ+½)−ψ(γ) ≈ 1/(2γ)  →  E ≈ 2·log 2  (= ell_vg(0), prior mode)

    Returns M · E_per  (total over all players).
    """
    gamma = pp["gamma"]
    E_per = (
        2.0 * jnp.log(2.0)
        - jax.scipy.special.digamma(gamma)
        + jax.scipy.special.digamma(gamma + 0.5)
    )
    return M * E_per


def neg_entropy_prior_binary(pp, M):
    """∫ p(θ)·log p(θ) dθ  for the binary VG prior  (= −H[p], negative entropy).

    For the normalised prior  p(θ) = p̃(θ)/Z_per  with p̃(θ) = σ(θ/s)^γ·σ(−θ/s)^γ:

        log p(θ) = −γ·ell_vg(θ/s) − log Z_per

    Taking the expectation under p itself:

        ∫ p·log p dθ = −γ·E_prior[ell_vg] − log Z_per

    Substituting E_prior[ell_vg] = −d log Z_per/dγ  and expanding log Z_per:

        = γ·(d log Z_per/dγ) − log Z_per
        = γ·(−2·log 2 + ψ(γ) − ψ(γ+½))
          − [log s + (1−2γ)·log 2 + ½·log π + lgamma(γ) − lgamma(γ+½)]

    Collecting terms (the −2γ·log 2 and +2γ·log 2 cancel):

        per player = γ·ψ(γ) − γ·ψ(γ+½) − log s − log 2 − ½·log π
                     − lgamma(γ) + lgamma(γ+½)

    Returns M · (per-player value).  Always ≤ 0 (entropy ≥ 0).
    Limiting behaviour:
        γ → 0 :  ≈ M·(log γ − log(2s) − 1)   → −∞  (diffuse prior, high entropy)
        γ → ∞ :  ≈ −M·γ·log 2  (prior concentrates at θ=0, entropy → −∞ too)
    """
    gamma = pp["gamma"]
    scale = pp["scale"]
    neg_h_per = (
        gamma * jax.scipy.special.digamma(gamma)
        - gamma * jax.scipy.special.digamma(gamma + 0.5)
        - jnp.log(scale)
        - jnp.log(2.0)
        - 0.5 * jnp.log(jnp.pi)
        - jax.scipy.special.gammaln(gamma)
        + jax.scipy.special.gammaln(gamma + 0.5)
    )
    return M * neg_h_per


def log_Z_gl(pp, M):
    """Exact log normalizer for the virtual-games prior via Gauss-Laguerre quadrature.

    Works for any L; reduces to log_Z_binary for L=2 (verified analytically).

    Derivation
    ----------
    Z_per = s · ∫ exp(−γ · ell_vg(θ/s)) dθ

    Split at θ=0 and substitute u = γ|θ|/s on each half:

        Z_per = (s/γ) · ∫₀^∞ exp(−u) · exp(h₊(u)) du
              + (s/γ) · ∫₀^∞ exp(−u) · exp(h₋(u)) du

    where
        h₊(u) = u − γ · ell_vg( u/γ)     (positive half, θ = s·u/γ)
        h₋(u) = u − γ · ell_vg(−u/γ)     (negative half, θ = −s·u/γ)

    Both integrands are bounded: for large u, ell_vg(u/γ) → u/γ + c_∞ so
    h₊(u) → −γ·c_∞ (constant).  The exp(−u) weight decays fast, so the
    Gauss-Laguerre rule ∫₀^∞ e^{−u} f(u) du ≈ Σ wᵢ f(uᵢ) converges quickly.

    In log-space (numerically stable):

        log Z_per = log(s) − log(γ)
                  + logsumexp([log wᵢ + h₊(uᵢ)] ∪ [log wᵢ + h₋(uᵢ)])

    Verification for L=2, γ=s=1
    ----------------------------
        ell_vg(z) = log(1+e^{−z}) + log(1+e^z)
        h₊(u)     = −2 log(1 + e^{−u})
        ∫₀^∞ e^{−u} exp(h₊) du = ∫₀^∞ e^{−u} σ(u)² du = ½    (Beta integral)
        h₊ = h₋  (symmetry)
        log Z_per = log(2) + log(½ + ½) = log(2) + log(1) ...

    Rechecking: logsumexp([log w + h₊] ∪ [log w + h₋]) = log(2·½) = 0,
    so log Z_per = 0  →  Z_per = 1.  Exact answer for L=2 at γ=s=1. ✓
    """
    gamma = pp["gamma"]
    scale = pp["scale"]

    def _ell_vg_scalar(z):
        """ell_vg for a single player at scaled strength z = θ/s."""
        v    = _ac_logits(z, pp)
        logZ = jax.scipy.special.logsumexp(v)
        return (logZ - v[-1]) + (logZ - v[0])

    # h₊(uᵢ) and h₋(uᵢ) over the 20 GL nodes
    h_plus  = vmap(lambda u: u - gamma * _ell_vg_scalar( u / gamma))(_GL_NODES)
    h_minus = vmap(lambda u: u - gamma * _ell_vg_scalar(-u / gamma))(_GL_NODES)

    log_w        = jnp.log(_GL_WEIGHTS)
    log_integral = jax.scipy.special.logsumexp(
        jnp.concatenate([log_w + h_plus, log_w + h_minus])
    )
    log_Z_per = jnp.log(scale) - jnp.log(gamma) + log_integral
    return M * log_Z_per


def log_Z_laplace(pp, M):
    """Log normalizer for the virtual-games prior via Laplace approximation at θ=0.

    Kept for reference / debugging.  Prefer log_Z_vg (exact for L=2,
    Gauss-Laguerre for L>2) in all production code.  The Laplace approximation
    gives log Z_per ~ −½ log γ instead of the correct −log γ for small γ.
    """
    gamma      = pp["gamma"]
    scale      = pp["scale"]
    v0         = _ac_logits(0.0, pp)                          # logits at z=0
    log_Z0     = jax.scipy.special.logsumexp(v0)
    log_f0     = gamma * ((v0[-1] - log_Z0) + (v0[0] - log_Z0))
    probs_0    = jax.nn.softmax(v0)
    delta_full = pp["Adelta"] @ pp["delta"] + pp["bdelta"]
    E0_delta   = jnp.sum(delta_full * probs_0)
    G_dot_0    = jnp.sum(delta_full ** 2 * probs_0) - E0_delta ** 2
    sigma_q    = scale / jnp.sqrt(2.0 * gamma * G_dot_0)
    log_Z_per  = log_f0 + 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(sigma_q)
    return M * log_Z_per


def _optimize_gamma_U(pp, data, ell_fn, reg_fn, log_Z_fn):
    """Find γ that minimises U_obj via bounded scalar search over log γ."""
    def u_of_loggamma(log_gamma):
        pp_g = {**pp, "gamma": float(jnp.exp(jnp.asarray(log_gamma)))}
        return U_obj(pp_g, data, ell_fn=ell_fn, reg_fn=reg_fn, log_Z_fn=log_Z_fn)
    res = scipy.optimize.minimize_scalar(
        u_of_loggamma,
        bounds=(-13.8, 6.9),   # log(1e-6) to log(1e3)
        method="bounded",
    )
    return float(jnp.exp(jnp.asarray(res.x)))


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatched loss / validation wrappers
# ─────────────────────────────────────────────────────────────────────────────
# _loss_fun(ell_fn, z, y, xi, hfa, params) — generic; ell_fn is static.
# loss_fun(z, y, xi, hfa, params)           — AC alias (backwards compatible).
# ═══════════════════════════════════════════════════════════════════════════════

@partial(jit, static_argnums=(0,))
def _loss_fun(ell_fn, z, y, xi, hfa, params):
    """Per-game weighted loss: apply scale + HFA, call ell_fn, weight by xi."""
    z_s = z / params["scale"] + hfa
    ell = ell_fn(z_s, y, params)
    return jnp.dot(xi, ell)


@partial(jit, static_argnums=(0,))
def _validation_fun(ell_fn, z, y, xi, hfa, params):
    """Per-game validation loss (identical to _loss_fun; kept separate for future use)."""
    z_s = z / params["scale"] + hfa
    ell = ell_fn(z_s, y, params)
    return jnp.dot(xi, ell)


# AC aliases
loss_fun       = partial(_loss_fun,       ac_ell)
validation_fun = partial(_validation_fun, ac_ell)


# ═══════════════════════════════════════════════════════════════════════════════
# Core objective J_obj and variants
# ═══════════════════════════════════════════════════════════════════════════════

@partial(jit, static_argnums=(0, 1))
def _J_obj(ell_fn, reg_fn, theta, data, pp, t_keep=None):
    """Unnormalised negative log-joint: Σ_t ell_t + reg_fn(θ, pp).

    Under the unnormalised convention H = ∇²_θ J_obj is the exact Hessian of the
    negative log-joint, so diag(H⁻¹) is the Laplace posterior variance directly
    (no 1/T factors).  When t_keep selects T-1 games, J_obj is the exact
    leave-one-out training objective.
    """
    X, y, u = data
    xi  = pp["weight"][u["category"]]
    hfa = u["hfa"] * pp["eta"]
    z   = (theta[u["home_idx"]] - theta[u["away_idx"]]
           if "home_idx" in u else X.T @ theta)
    loss = vmap(partial(_loss_fun, ell_fn), in_axes=[0, 0, 0, 0, None])(z, y, xi, hfa, pp)
    return jnp.sum(loss[t_keep]) + reg_fn(theta, pp)


@partial(jit, static_argnums=(0, 1))
def _J_obj_partial(ell_fn, reg_fn, pp_var, pp_const, data, theta_hat_fix):
    """Compiled-once outer objective for optimize_ML (theta frozen)."""
    return _J_obj(ell_fn, reg_fn, theta_hat_fix, data, dict(**pp_var, **pp_const))


# AC aliases (backwards compatible)
J_obj         = partial(_J_obj,         ac_ell, regularization_fun)
J_obj_partial = partial(_J_obj_partial, ac_ell, regularization_fun)


# ── Virtual-games regularisation aliases ─────────────────────────────────────
J_obj_vg         = partial(_J_obj,         ac_ell, regularization_fun_vg)
J_obj_vg_partial = partial(_J_obj_partial, ac_ell, regularization_fun_vg)


# ═══════════════════════════════════════════════════════════════════════════════
# theta_hat  —  MAP estimate of team strengths (inner loop)
# ═══════════════════════════════════════════════════════════════════════════════

@partial(jit, static_argnums=(0, 1))
def _theta_hat(ell_fn, reg_fn, pp, data, t_keep=None):
    X, _, _ = data
    M = X.shape[0]
    solver = jaxopt.BFGS(
        fun=partial(_J_obj, ell_fn, reg_fn), maxiter=200, implicit_diff=True, verbose=False
    )
    res = solver.run(jnp.zeros(M), data, pp, t_keep)
    return res.params


theta_hat    = partial(_theta_hat, ac_ell, regularization_fun)
theta_hat_vg = partial(_theta_hat, ac_ell, regularization_fun_vg)


# ═══════════════════════════════════════════════════════════════════════════════
# Marginal ML — Laplace evidence U_obj and importance-sampling variant
# ═══════════════════════════════════════════════════════════════════════════════

def U_obj(pp, data, ell_fn=None, reg_fn=None, log_Z_fn=None):
    """Marginal objective (Laplace evidence):  -log p(y | alpha, delta, gamma).

    Under the unnormalised convention:

        U = J_obj(θ̂) + log_Z_fn(pp, M) + ½·log det H

    where H = ∇²_θ J_obj(θ̂) is the exact Hessian and log_Z_fn returns the
    log normalizer of the unnormalised prior (see log_Z_ridge, log_Z_laplace).

    ell_fn   : elementary loss (default: ac_ell).
    reg_fn   : regularization penalty (default: L2/ridge).
    log_Z_fn : log normalizer of the prior (default: log_Z_ridge).
    """
    if ell_fn is None:
        ell_fn = ac_ell
    if reg_fn is None:
        reg_fn = regularization_fun
    if log_Z_fn is None:
        log_Z_fn = log_Z_ridge
    _J  = partial(_J_obj, ell_fn, reg_fn)
    _th = partial(_theta_hat, ell_fn, reg_fn)

    X, y, u = data
    M = X.shape[0]
    th = _th(pp, data)

    J_val      = float(_J(th, data, pp))
    prior_norm = log_Z_fn(pp, M)
    H       = jax.hessian(_J)(th, data, pp)
    # Clamp near-zero eigenvalues before computing log det.
    # Needed when the VG prior's Laplace tails give vanishing curvature at
    # extreme θ̂ (e.g. a team with all wins/losses in an L=2 dataset).
    eigvals  = jnp.linalg.eigvalsh(H)
    thresh   = jnp.abs(eigvals).max() * 1e-8
    log_det  = jnp.sum(jnp.log(jnp.maximum(eigvals, thresh)))
    return float(J_val + prior_norm + 0.5 * float(log_det))


def U_obj_IS(pp, data, n_samples=2000, seed=0, ell_fn=None, reg_fn=None):
    """Importance-sampling estimate of -log p(y | pp).

    Proposal: N(θ̂, H⁻¹) (Laplace approximation to the posterior).
    When the posterior is exactly Gaussian, IS == Laplace and ESS/N = 1.

    Returns
    -------
    (float, float) : (-log p̂(y|pp),  relative ESS ∈ (0, 1])
    """
    import jax.random as jr
    if ell_fn is None:
        ell_fn = ac_ell
    if reg_fn is None:
        reg_fn = regularization_fun
    _J  = partial(_J_obj, ell_fn, reg_fn)
    _th = partial(_theta_hat, ell_fn, reg_fn)

    X, y, u = data
    M = X.shape[0]
    c = float(pp["gamma"])

    th_hat       = _th(pp, data)
    H            = jax.hessian(_J)(th_hat, data, pp)
    _, log_det_H = jnp.linalg.slogdet(H)
    H_inv        = jnp.linalg.inv(H)
    L_chol       = jnp.linalg.cholesky(H_inv)

    key  = jr.PRNGKey(seed)
    z    = jr.normal(key, (n_samples, M))
    samp = th_hat[None, :] + z @ L_chol.T          # (N, M)

    J_vals = jax.vmap(lambda th: _J(th, data, pp))(samp)   # (N,)
    diff   = samp - th_hat[None, :]                          # (N, M)
    quad   = 0.5 * jnp.einsum('ni,ij,nj->n', diff, H, diff) # (N,)
    log_w  = -J_vals + 0.5 * M * jnp.log(c) - 0.5 * log_det_H + quad

    log_p   = jax.nn.logsumexp(log_w) - jnp.log(float(n_samples))
    log_ess = 2.0 * jax.nn.logsumexp(log_w) - jax.nn.logsumexp(2.0 * log_w)
    ess_rel = float(jnp.exp(log_ess)) / n_samples

    return float(-log_p), ess_rel


# ═══════════════════════════════════════════════════════════════════════════════
# EM — Q_marginal (GH quadrature) and optimize_EM
# ═══════════════════════════════════════════════════════════════════════════════

@partial(jit, static_argnums=(0,))
def _Q_marginal(ell_fn, pp_var, pp_const, y, xi_t, u_hfa, z_nodes):
    """EM Q-function: expected complete-data log-likelihood via GH quadrature.

    JAX-differentiable w.r.t. any parameters in pp_var.
    Uses module-level GH constants _GH_NODES / _GH_WEIGHTS.

    Parameters
    ----------
    pp_var   : dict          free parameters (optimisation variable)
    pp_const : dict          frozen parameters
    y        : (T,)          integer outcome labels
    xi_t     : (T,)          per-game category weight
    u_hfa    : (T,)          raw HFA vector; eta read from pp so gradient flows
    z_nodes  : (T, _GH_N)   frozen GH quadrature nodes from E-step
    """
    pp    = {**pp_const, **pp_var}
    hfa_t = u_hfa * pp["eta"]

    def game_sum(z_t_nodes, y_t, xi_val, hfa_val):
        losses = jax.vmap(
            lambda z: _loss_fun(ell_fn, z, y_t, xi_val, hfa_val, pp)
        )(z_t_nodes)
        return jnp.dot(_GH_WEIGHTS, losses)

    per_game = jax.vmap(game_sum)(z_nodes, y, xi_t, hfa_t)   # (T,)
    return jnp.sum(per_game) / jnp.sqrt(jnp.pi)


Q_marginal = partial(_Q_marginal, ac_ell)


def optimize_EM(pp_init, data, opt_scheduling, n_em_iter=30, tol=1e-6,
                ell_fn=None, reg_fn=None, log_Z_fn=None, closed_form_gamma=True):
    """EM optimizer for marginal ML.

    E-step : MAP θ̂ + inverse Hessian H⁻¹ (Laplace approximation).
    M-step : γ update (closed form for ridge, scalar U_obj search otherwise);
             joint L-BFGS on Q_marginal for all other params.

    opt_scheduling follows the list-of-dicts convention used by optimize_ML /
    optimize_ALO.  Each dict maps param_name → mask.  The mask content is not
    used (all elements are free); it is retained for interface consistency.

    Parameters
    ----------
    pp_init            : dict or list of dicts
    data               : (X, y, u)
    opt_scheduling     : list of dicts  {param_name: mask}
    n_em_iter          : max EM iterations per pp
    tol                : convergence threshold on max(|Δparam|, |Δγ|/γ)
    ell_fn             : elementary loss (default: ac_ell)
    reg_fn             : regularization penalty (default: L2/ridge)
    log_Z_fn           : negative log normalizer of the prior (default: log_Z_ridge)
    closed_form_gamma  : True → closed-form γ update (L2 only); False → scalar
                         minimisation of U_obj over log γ

    Returns
    -------
    dict or list of dicts (matches input type)
    """
    if ell_fn is None:
        ell_fn = ac_ell
    if reg_fn is None:
        reg_fn = regularization_fun
    if log_Z_fn is None:
        log_Z_fn = log_Z_ridge
    _th = partial(_theta_hat, ell_fn, reg_fn)
    _J  = partial(_J_obj,     ell_fn, reg_fn)
    _Q  = partial(_Q_marginal, ell_fn)

    X, y, u = data
    hi = jnp.array(u["home_idx"])
    ai = jnp.array(u["away_idx"])

    input_is_LIST = isinstance(pp_init, list)
    if not input_is_LIST:
        pp_init = [pp_init]

    all_em_keys = set()
    for block in opt_scheduling:
        all_em_keys.update(block.keys())
    has_gamma = "gamma" in all_em_keys

    nongamma_blocks = [{k: v for k, v in block.items() if k != "gamma"}
                       for block in opt_scheduling]
    nongamma_blocks = [b for b in nongamma_blocks if b]

    pp_out_list = []
    for pp_k in pp_init:
        pp_cur = copy.deepcopy(pp_k)

        for _ in range(n_em_iter):
            pp_prev = copy.deepcopy(pp_cur)

            # ── E-step ────────────────────────────────────────────────────────
            th    = _th(pp_cur, data)
            H     = jax.hessian(_J)(th, data, pp_cur)
            H_inv = jnp.linalg.inv(H)

            z_t  = th[hi] - th[ai]
            v_t  = jnp.maximum(
                H_inv[hi, hi] + H_inv[ai, ai] - 2.0 * H_inv[hi, ai], 1e-12)
            z_nodes = (jnp.sqrt(2.0 * v_t[:, None]) * _GH_NODES[None, :]
                       + z_t[:, None])                        # (T, _GH_N)

            xi_t  = jnp.array(pp_cur["weight"][u["category"]])
            u_hfa = jnp.array(u["hfa"], dtype=float)

            # ── M-step γ ──────────────────────────────────────────────────
            if has_gamma:
                if closed_form_gamma:
                    theta2 = jnp.diag(H_inv) + th ** 2       # posterior 2nd moments
                    pp_cur["gamma"] = float(1.0 / jnp.mean(theta2))
                else:
                    pp_cur["gamma"] = _optimize_gamma_U(
                        pp_cur, data, ell_fn, reg_fn, log_Z_fn)

            # ── M-step other params (BFGS on Q) ───────────────────────────
            for block in nongamma_blocks:
                pp_var   = {k: pp_cur[k] for k in block}
                pp_const = {k: v for k, v in pp_cur.items() if k not in block}
                solver   = jaxopt.BFGS(fun=_Q, maxiter=200, tol=1e-8, implicit_diff=False)
                res      = solver.run(pp_var, pp_const, y, xi_t, u_hfa, z_nodes)
                pp_cur.update(res.params)

            # ── Convergence ────────────────────────────────────────────────
            delta_max = 0.0
            for key in all_em_keys:
                if key == "gamma":
                    delta_max = max(delta_max,
                        abs(pp_cur["gamma"] - pp_prev["gamma"])
                        / (abs(pp_prev["gamma"]) + 1e-12))
                else:
                    delta_max = max(delta_max, float(jnp.max(jnp.abs(
                        jnp.array(pp_cur[key]) - jnp.array(pp_prev[key])))))
            if delta_max < tol:
                break

        pp_out_list.append(copy.deepcopy(pp_cur))

    return pp_out_list if input_is_LIST else pp_out_list[0]


def marginal_uncertainty(pp_opt, data, free_keys, ell_fn=None, reg_fn=None, log_Z_fn=None):
    """Posterior std and covariance of free EM parameters (Laplace).

    Uses the envelope theorem: at the MAP, dJ/dθ = 0, so the total derivative
    of U w.r.t. any free parameter equals the partial derivative holding θ fixed.

    γ is reparameterised as g = log(γ) for the Laplace approximation
    (unconstrained, more symmetric posterior).

    Parameters
    ----------
    pp_opt    : converged parameter dict (from optimize_EM)
    data      : (X, y, u)
    free_keys : list[str]   e.g. ["alpha", "eta", "gamma"]
    ell_fn    : elementary loss (default: ac_ell)
    reg_fn    : regularization penalty (default: L2/ridge)
    log_Z_fn  : negative log normalizer of the prior (default: log_Z_ridge)

    Returns
    -------
    std_dict   : {k: np.ndarray}  posterior std per free parameter
    cov        : (N, N) np.ndarray  full posterior covariance
    key_slices : {k: slice}  position of each key in the flat vector
    """
    if ell_fn is None:
        ell_fn = ac_ell
    if reg_fn is None:
        reg_fn = regularization_fun
    if log_Z_fn is None:
        log_Z_fn = log_Z_ridge
    _th = partial(_theta_hat, ell_fn, reg_fn)
    _J  = partial(_J_obj,     ell_fn, reg_fn)

    X, y, u = data
    M = X.shape[0]

    _LOG_KEYS = {"gamma"}
    free_dict = {
        k: jnp.log(jnp.asarray(pp_opt[k])) if k in _LOG_KEYS
        else jnp.asarray(pp_opt[k])
        for k in free_keys
    }

    N_free = sum(max(int(jnp.asarray(pp_opt[k]).size), 1) for k in free_keys)
    if N_free == 0:
        return {k: np.zeros(0) for k in free_keys}, np.zeros((0, 0)), {}

    th_fixed = _th(pp_opt, data)    # frozen — not differentiated (envelope theorem)

    def U_frozen(free_dict):
        transformed = {
            k: jnp.exp(v) if k in _LOG_KEYS else v
            for k, v in free_dict.items()
        }
        pp         = {**pp_opt, **transformed}
        J_val      = _J(th_fixed, data, pp)
        prior_norm = log_Z_fn(pp, M)
        H_theta    = jax.hessian(_J)(th_fixed, data, pp)
        jitter     = 0 * jnp.trace(H_theta) / M * 1e-8
        H_stab     = H_theta + jitter * jnp.eye(M)
        _, log_det = jnp.linalg.slogdet(H_stab)
        return J_val + prior_norm + 0.5 * log_det

    gg = jax.grad(U_frozen)(free_dict)
    HH = jax.hessian(U_frozen)(free_dict)

    sch = {k: np.ones(max(int(jnp.asarray(pp_opt[k]).size), 1), int)
           for k in free_keys}
    _, H_mat = dict2num(sch, gg, HH)

    cov      = np.array(np.linalg.inv(H_mat))
    std_flat = np.sqrt(np.maximum(np.diag(cov), 0.0))

    std_dict   = {}
    key_slices = {}
    pos = 0
    for k in free_keys:
        size_k        = max(int(jnp.asarray(pp_opt[k]).size), 1)
        sl            = slice(pos, pos + size_k)
        std_dict[k]   = std_flat[sl]
        key_slices[k] = sl
        pos          += size_k

    return std_dict, cov, key_slices


# ═══════════════════════════════════════════════════════════════════════════════
# dict2num  — gradient/Hessian dict → flat vector/matrix  (used by optimize_ALO)
# ═══════════════════════════════════════════════════════════════════════════════

def dict2num(sch, gg, HH):
    """Flatten gradient dict gg and Hessian dict HH into 1-D/2-D arrays.

    sch : {param_name: mask_array}  — elements with mask==1 are selected.
    """
    KK = sum(int(np.sum(sch[e])) for e in sch)
    grad_ALO = np.zeros((KK,))
    Hess_ALO = np.zeros((KK, KK))
    k_start = 0
    for elem in sch:
        bb    = sch[elem] == np.ones_like(sch[elem])
        k_end = k_start + int((bb * 1).sum())
        grad_ALO[k_start:k_end] = gg[elem].reshape(len(bb),)[bb]
        k_start2 = 0
        for elem2 in sch:
            bb2    = sch[elem2] == np.ones_like(sch[elem2])
            k_end2 = k_start2 + int((bb2 * 1).sum())
            Hess_ALO[k_start:k_end, k_start2:k_end2] = (
                HH[elem][elem2].reshape((len(bb), len(bb2)))[bb][:, bb2])
            k_start2 = k_end2
        k_start = k_end
    return grad_ALO, Hess_ALO


# ═══════════════════════════════════════════════════════════════════════════════
# ML optimisation
# ═══════════════════════════════════════════════════════════════════════════════

def optimize_ML(pp_init, data, opt_scheduling, obj_partial=None, ell_fn=None, reg_fn=None):
    """ML optimisation: alternate θ (inner BFGS) and shared params (outer BFGS).

    Parameters
    ----------
    pp_init        : dict or list of dicts
    data           : (X, y, u)
    opt_scheduling : list of dicts  {param_name: mask}
    obj_partial    : outer objective (default: _J_obj_partial bound to ell_fn, reg_fn).
    ell_fn         : elementary loss (default: ac_ell)
    reg_fn         : regularization penalty (default: L2/ridge)

    Returns
    -------
    (pp_out, theta_hat_out) — pp_out matches list/dict type of pp_init
    """
    if ell_fn is None:
        ell_fn = ac_ell
    if reg_fn is None:
        reg_fn = regularization_fun
    if obj_partial is None:
        obj_partial = partial(_J_obj_partial, ell_fn, reg_fn)

    _th = partial(_theta_hat, ell_fn, reg_fn)
    _lf = partial(_loss_fun,  ell_fn)

    X, y, u = data
    M, T    = X.shape
    eps_precision = 1e-6

    input_is_LIST = isinstance(pp_init, list)
    if not input_is_LIST:
        pp_init = [pp_init]

    maxrounds    = 20
    pp_out_list  = []
    theta_hat_out = jnp.zeros(M)
    for pp_init_k in pp_init:
        pp_out = copy.deepcopy(pp_init_k)
        J_ref  = np.inf

        for i_round in range(maxrounds):
            for sch in opt_scheduling:
                theta_hat_out = _th(pp_out, data)
                pp_var   = {key: pp_out[key] for key in sch}
                pp_const = {key: pp_out[key]
                            for key in set(pp_out.keys()).difference(pp_var.keys())}
                solver   = jaxopt.BFGS(
                    fun=obj_partial, maxiter=200, implicit_diff=False, verbose=False)
                res      = solver.run(pp_var, pp_const, data, theta_hat_out)
                pp_out.update(res.params)

            z_hat = (theta_hat_out[u["home_idx"]] - theta_hat_out[u["away_idx"]]
                     if "home_idx" in u else X.T @ theta_hat_out)
            hfa   = u["hfa"] * pp_out["eta"]
            xi    = pp_out["weight"][u["category"]]
            pred  = vmap(_lf, in_axes=[0, 0, 0, 0, None])(z_hat, y, xi, hfa, pp_out)
            J_ref_old = J_ref
            J_ref     = float(pred.mean())
            if np.abs(J_ref - J_ref_old) < eps_precision * J_ref:
                break

        pp_out_list.append(copy.deepcopy(pp_out))

    pp_out = pp_out_list if input_is_LIST else pp_out_list[0]
    return pp_out, theta_hat_out


# ═══════════════════════════════════════════════════════════════════════════════
# ALO (Approximate Leave-One-Out) and optimize_ALO
# ═══════════════════════════════════════════════════════════════════════════════

@partial(jit, static_argnums=(0, 1))
def _ALO(ell_fn, reg_fn, pp, data):
    """Approximate leave-one-out cross-validation via Sherman-Morrison formula.

    Returns (mean_alo, per_game_alo).
    """
    X, y, u = data
    xi  = pp["weight"][u["category"]]
    hfa = u["hfa"] * pp["eta"]
    M, T = X.shape

    theta_hat_out = _theta_hat(ell_fn, reg_fn, pp, data)
    H     = hessian(partial(_J_obj, ell_fn, reg_fn))(theta_hat_out, data, pp)
    H_inv = jax.scipy.linalg.inv(H)

    if "home_idx" in u:
        home_idx, away_idx = u["home_idx"], u["away_idx"]
        a     = (H_inv[home_idx, home_idx] + H_inv[away_idx, away_idx]
                 - 2 * H_inv[home_idx, away_idx])
        z_hat = theta_hat_out[home_idx] - theta_hat_out[away_idx]
    else:
        a     = jnp.sum(jnp.multiply(H_inv @ X, X), axis=0)
        z_hat = X.T @ theta_hat_out

    _lf      = partial(_loss_fun,       ell_fn)
    _vf      = partial(_validation_fun, ell_fn)
    ell_dot  = vmap(grad(_lf),    in_axes=[0, 0, 0, 0, None])(z_hat, y, xi, hfa, pp)
    ell_ddot = vmap(hessian(_lf), in_axes=[0, 0, 0, 0, None])(z_hat, y, xi, hfa, pp)

    z_hat_approx = z_hat + ell_dot * a / (1 - ell_ddot * a)
    pred = vmap(_vf, in_axes=[0, 0, 0, 0, None])(z_hat_approx, y, xi, hfa, pp)

    return jnp.sum(pred) / len(y), pred


ALO = partial(_ALO, ac_ell, regularization_fun)


@partial(jit, static_argnums=(0, 1))
def _ALO_marginal(ell_fn, reg_fn, pp, data):
    """ALO with GH-marginalised predictive: accounts for posterior uncertainty in z.

    Add-on to _ALO — the existing function is unchanged.  The LOO predictive
    log-loss for game t is

        −log ∫ P(Y=y_t | z) N(z ; z_loo_t, v_loo_t) dz

    approximated by Gauss-Hermite quadrature (same _GH_NODES / _GH_WEIGHTS as
    the EM M-step).

    LOO sufficient statistics (Sherman-Morrison algebra):

        a_t     = x_t^T H⁻¹ x_t            (full-data posterior variance of z_t)
        z_loo_t = z_hat_t + ℓ̇_t a_t / (1 − ℓ̈_t a_t)    (LOO mean — identical to _ALO)
        v_loo_t = a_t / (1 − ℓ̈_t a_t)                    (LOO variance — new here)

    Limiting behaviour:
      v_loo → 0  ⟹  GH integral collapses to the point mass at z_loo
               ⟹  returns the same value as _ALO.

    Returns (mean_alo_marginal, per_game_alo_marginal).
    """
    X, y, u = data
    xi  = pp["weight"][u["category"]]
    hfa = u["hfa"] * pp["eta"]
    M, T = X.shape

    theta_hat_out = _theta_hat(ell_fn, reg_fn, pp, data)
    H     = hessian(partial(_J_obj, ell_fn, reg_fn))(theta_hat_out, data, pp)
    H_inv = jax.scipy.linalg.inv(H)

    if "home_idx" in u:
        home_idx, away_idx = u["home_idx"], u["away_idx"]
        a     = (H_inv[home_idx, home_idx] + H_inv[away_idx, away_idx]
                 - 2 * H_inv[home_idx, away_idx])
        z_hat = theta_hat_out[home_idx] - theta_hat_out[away_idx]
    else:
        a     = jnp.sum(jnp.multiply(H_inv @ X, X), axis=0)
        z_hat = X.T @ theta_hat_out

    _lf      = partial(_loss_fun, ell_fn)
    ell_dot  = vmap(grad(_lf),    in_axes=[0, 0, 0, 0, None])(z_hat, y, xi, hfa, pp)
    ell_ddot = vmap(hessian(_lf), in_axes=[0, 0, 0, 0, None])(z_hat, y, xi, hfa, pp)

    denom = 1.0 - ell_ddot * a
    z_loo = z_hat + ell_dot * a / denom
    v_loo = jnp.maximum(a / denom, 1e-12)          # LOO posterior variance of z_t

    # GH nodes: z_nodes[t, k] = z_loo[t] + sqrt(2 v_loo[t]) · ξ_k   shape (T, GH_N)
    z_nodes = z_loo[:, None] + jnp.sqrt(2.0 * v_loo[:, None]) * _GH_NODES[None, :]

    log_w = jnp.log(_GH_WEIGHTS)                    # (GH_N,)  — GH weights are positive

    def _game_loss(z_t_nodes, y_t, xi_val, hfa_val):
        """Marginalised log-loss for one game.

        −log(1/√π · Σ_k w_k exp(−ℓ_k))
          = ½ log π − logsumexp_k(log w_k − ℓ_k)
        """
        node_losses = vmap(lambda z_n: _lf(z_n, y_t, xi_val, hfa_val, pp))(z_t_nodes)
        return 0.5 * jnp.log(jnp.pi) - jax.nn.logsumexp(log_w - node_losses)

    pred = vmap(_game_loss)(z_nodes, y, xi, hfa)    # (T,)
    return jnp.sum(pred) / len(y), pred


ALO_marginal = partial(_ALO_marginal, ac_ell, regularization_fun)


def optimize_ALO(pp_init, data, opt_scheduling, ell_fn=None, reg_fn=None):
    """ALO Newton optimiser for (alpha, delta, eta).

    ell_fn : elementary loss (default: ac_ell).
    reg_fn : regularization penalty (default: L2/ridge).
    """
    if ell_fn is None:
        ell_fn = ac_ell
    if reg_fn is None:
        reg_fn = regularization_fun
    _alo = partial(_ALO, ell_fn, reg_fn)

    @jit
    def ALO_partial(pp_var, pp_const, data):
        return _alo(dict(**pp_var, **pp_const), data)

    input_is_LIST = isinstance(pp_init, list)
    if not input_is_LIST:
        pp_init = [pp_init]

    maxrounds   = 10
    pp_out_list = []
    for pp_init_k in pp_init:
        pp_out = copy.deepcopy(pp_init_k)
        J_ref  = np.inf

        for i_round in range(maxrounds):
            J_ref_old = J_ref
            J_ref     = _alo(pp_out, data)
            HAS_AUX   = isinstance(J_ref, tuple)
            if HAS_AUX:
                J_ref = J_ref[0]
            if np.abs(J_ref - J_ref_old) < 1e-6 * abs(J_ref):
                break

            for sch in opt_scheduling:
                pp_var   = {key: pp_out[key] for key in sch}
                pp_const = {key: pp_out[key]
                            for key in set(pp_out.keys()).difference(pp_var.keys())}
                gg = grad(ALO_partial, has_aux=HAS_AUX)(pp_var, pp_const, data)
                HH = jax.jacrev(
                    jax.jacrev(ALO_partial, has_aux=HAS_AUX), has_aux=HAS_AUX
                )(pp_var, pp_const, data)
                if HAS_AUX:
                    gg, HH = gg[0], HH[0]

                grad_ALO, Hess_ALO = dict2num(sch=sch, gg=gg, HH=HH)
                min_eig = np.linalg.eigvalsh(Hess_ALO).min()
                if min_eig < 1e-6:
                    Hess_ALO += (1e-6 - min_eig) * np.eye(len(grad_ALO))
                d_pp    = jax.scipy.linalg.solve(Hess_ALO, grad_ALO, assume_a="her")

                k_start = 0
                for elem in sch:
                    bb    = sch[elem] == np.ones_like(sch[elem])
                    L_bb  = int((bb * 1).sum())
                    k_end = k_start + L_bb
                    if L_bb == 1:
                        pp_out[elem] -= d_pp[k_start]
                    else:
                        pp_out[elem][bb] -= d_pp[k_start:k_end]
                    k_start = k_end

        pp_out_list.append(copy.deepcopy(pp_out))

    return pp_out_list if input_is_LIST else pp_out_list[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def v_theta_fun(pp, data, ell_fn=None, reg_fn=None):
    """Non-sandwich and sandwich variance of θ̂.

    v_ns[m] = H⁻¹_mm       (Laplace posterior variance)
    v_sw[m] = ||C_{m,:} * g||²   where C = H⁻¹X, g = ∇_z ell_t at fitted z.

    ell_fn : elementary loss (default: ac_ell).
    reg_fn : regularization penalty (default: L2/ridge).
    """
    if ell_fn is None:
        ell_fn = ac_ell
    if reg_fn is None:
        reg_fn = regularization_fun
    _J  = partial(_J_obj,    ell_fn, reg_fn)
    _lf = partial(_loss_fun, ell_fn)

    X, y, u = data
    xi  = pp["weight"][u["category"]]
    hfa = u["hfa"] * pp["eta"]
    M, T = X.shape

    theta_hat_out = _theta_hat(ell_fn, reg_fn, pp, data)
    H     = hessian(_J)(theta_hat_out, data, pp)
    H_inv = jax.scipy.linalg.inv(H)

    z_hat = (theta_hat_out[u["home_idx"]] - theta_hat_out[u["away_idx"]]
             if "home_idx" in u else X.T @ theta_hat_out)
    g     = vmap(grad(_lf), in_axes=[0, 0, 0, 0, None])(z_hat, y, xi, hfa, pp)

    v_ns = jnp.diag(H_inv)
    if "home_idx" in u:
        C = H_inv[:, u["home_idx"]] - H_inv[:, u["away_idx"]]
    else:
        C = H_inv @ X
    v_sw = jnp.sum((C * g) ** 2, axis=1)

    return v_ns, v_sw


def GALO(pp, data, leave_out, ell_fn=None, reg_fn=None):
    """Generalised ALO: approximate leave-group-out cross-validation.

    leave_out : list of index arrays, one per group.
    Reduces to ALO when every group has exactly one game.

    ell_fn : elementary loss (default: ac_ell).
    reg_fn : regularization penalty (default: L2/ridge).
    """
    if ell_fn is None:
        ell_fn = ac_ell
    if reg_fn is None:
        reg_fn = regularization_fun
    _J  = partial(_J_obj,          ell_fn, reg_fn)
    _lf = partial(_loss_fun,       ell_fn)
    _vf = partial(_validation_fun, ell_fn)

    X, y, u = data
    xi  = pp["weight"][u["category"]]
    hfa = u["hfa"] * pp["eta"]
    M, T = X.shape

    theta_hat_out = _theta_hat(ell_fn, reg_fn, pp, data)
    H     = hessian(_J)(theta_hat_out, data, pp)
    H_inv = jax.scipy.linalg.inv(H)

    z_hat    = X.T @ theta_hat_out
    ell_dot  = vmap(grad(_lf),    in_axes=[0, 0, 0, 0, None])(z_hat, y, xi, hfa, pp)
    ell_ddot = vmap(hessian(_lf), in_axes=[0, 0, 0, 0, None])(z_hat, y, xi, hfa, pp)

    pred = 0.0
    for ii in leave_out:
        ii  = jnp.array(ii)
        k   = len(ii)
        A   = X[:, ii].T @ H_inv @ X[:, ii]
        z_approx = z_hat[ii] + A @ jnp.linalg.solve(
            jnp.eye(k) - jnp.diag(ell_ddot[ii]) @ A, ell_dot[ii])
        pred += vmap(_vf, in_axes=[0, 0, 0, 0, None])(
            z_approx, y[ii], xi[ii], hfa[ii], pp).mean()

    return float(pred) / len(leave_out)


def LOO(pp, data, ell_fn=None, reg_fn=None):
    """Exact LOO via t_keep mask (Sherman-Morrison approximation of the outer fit)."""
    if ell_fn is None:
        ell_fn = ac_ell
    if reg_fn is None:
        reg_fn = regularization_fun
    _th = partial(_theta_hat,    ell_fn, reg_fn)
    _vf = partial(_validation_fun, ell_fn)

    X, y, u = data
    xi  = pp["weight"][u["category"]]
    hfa = u["hfa"] * pp["eta"]
    M, T = X.shape

    pred    = jnp.zeros(T)
    tt_full = jnp.arange(T)
    for t in range(T):
        t_keep        = jnp.delete(tt_full, t)
        theta_hat_out = _th(pp, data, t_keep)
        z_hat         = jnp.dot(X[:, t], theta_hat_out)
        pred          = pred.at[t].set(_vf(z_hat, y[t], xi[t], hfa[t], pp))

    return jnp.mean(pred), pred


def LOO_marginal(pp, data, ell_fn=None, reg_fn=None):
    """Exact LOO with GH-marginalised predictive: accounts for LOO posterior uncertainty.

    Add-on to LOO — the existing function is unchanged.

    For each held-out game t:
      1. Refit θ on T−1 games (exact LOO MAP θ̂_{−t}).
      2. Compute the LOO Laplace posterior variance of z_t:
             v_t^{−t} = x_t^T H_{−t}^{−1} x_t
         where H_{−t} = ∇²_θ J_{−t}(θ̂_{−t}) is the LOO Hessian.
      3. Evaluate −log ∫ P(Y=y_t | z) N(z ; z_loo_t, v_t^{−t}) dz
         via Gauss-Hermite quadrature (same nodes as the EM M-step).

    Cost: T MAP fits + T Hessian inversions — same order as LOO.
    For M=20, T=380 this is fast (M is small; dominant cost is the T BFGS solves).

    Returns (mean_loo_marginal, per_game_loo_marginal).
    """
    if ell_fn is None:
        ell_fn = ac_ell
    if reg_fn is None:
        reg_fn = regularization_fun
    _th = partial(_theta_hat, ell_fn, reg_fn)
    _J  = partial(_J_obj,     ell_fn, reg_fn)
    _lf = partial(_loss_fun,  ell_fn)

    X, y, u = data
    xi  = pp["weight"][u["category"]]
    hfa = u["hfa"] * pp["eta"]
    M, T = X.shape

    log_w   = jnp.log(_GH_WEIGHTS)
    pred    = jnp.zeros(T)
    tt_full = jnp.arange(T)

    for t in range(T):
        t_keep    = jnp.delete(tt_full, t)
        theta_loo = _th(pp, data, t_keep)
        H_loo     = jax.hessian(_J)(theta_loo, data, pp, t_keep)
        H_loo_inv = jnp.linalg.inv(H_loo)

        x_t = X[:, t]
        z_t = jnp.dot(x_t, theta_loo)
        v_t = jnp.maximum(x_t @ H_loo_inv @ x_t, 1e-12)

        z_nodes     = z_t + jnp.sqrt(2.0 * v_t) * _GH_NODES        # (GH_N,)
        node_losses = vmap(lambda z_n: _lf(z_n, y[t], xi[t], hfa[t], pp))(z_nodes)
        pred = pred.at[t].set(
            0.5 * jnp.log(jnp.pi) - jax.nn.logsumexp(log_w - node_losses)
        )

    return jnp.mean(pred), pred


def LOO_explicit(pp, data, ell_fn=None, reg_fn=None):
    """Exact LOO by explicit dataset removal (reference implementation)."""
    if ell_fn is None:
        ell_fn = ac_ell
    if reg_fn is None:
        reg_fn = regularization_fun
    _th = partial(_theta_hat,    ell_fn, reg_fn)
    _vf = partial(_validation_fun, ell_fn)

    X, y, u  = data
    M, T     = X.shape
    xi       = pp["weight"][u["category"]]
    hfa      = u["hfa"] * pp["eta"]
    pred     = jnp.zeros(T)
    u_train  = {f: None for f in u}

    for t in range(T):
        X_train = jnp.delete(X, t, axis=1)
        y_train = jnp.delete(y, t)
        for f in u_train:
            u_train[f] = jnp.delete(u[f], t)
        theta_hat_out = _th(pp, (X_train, y_train, u_train))
        z_t  = jnp.dot(X[:, t], theta_hat_out)
        pred = pred.at[t].set(_vf(z_t, y[t], xi[t], hfa[t], pp))

    return jnp.mean(pred), pred


# ═══════════════════════════════════════════════════════════════════════════════
# make_fitter — convenience factory
# ═══════════════════════════════════════════════════════════════════════════════


# predefined log-likelihood functions
_ELL_FN_REGISTRY = {
    "AC":   ac_ell,
    "CL":   cl_ell,
    "FIVB": fivb_loss,
}

# predefined log-priors, normalizers; the latter are integrated over if closed_form_gamma=False
# Each entry: (reg_fn, log_Z_fn, closed_form_gamma)
_REG_FN_REGISTRY = {
    "L2":       (regularization_fun,    log_Z_ridge, True),
    "LOGISTIC": (regularization_fun_vg, log_Z_gl,       False),
}

def make_fitter(ell_fn="AC", reg_fn="L2"):
    """Return a SimpleNamespace of pre-bound fitting and prediction functions.

    ell_fn : str or callable
        "AC"   → ac_ell   (Adjusting Categories / softmax)
        "CL"   → cl_ell   (Cumulative Link / probit or logit)
        "FIVB" → fivb_loss
        or pass a callable directly.

    reg_fn : str or callable
        "L2"       → ridge / Gaussian prior  (closed-form γ M-step)
        "logistic" → virtual-games prior     (scalar U_obj search for γ)
        or pass a callable directly; in that case log_Z_fn defaults to
        log_Z_ridge and closed_form_gamma defaults to False.

    All functions in the returned namespace have ell_fn, reg_fn, log_Z_fn,
    and (where applicable) closed_form_gamma pre-bound, so no regularizer
    arguments are needed at call time::

        ac    = make_fitter()              # AC + ridge (defaults)
        ac_vg = make_fitter("AC", "logistic")

        ac_vg.optimize_EM(pp, data, sch)  # VG prior, γ via U_obj search
        ac_vg.U_obj(pp, data)             # VG evidence
        ac_vg.theta_hat(pp, data)         # MAP with VG prior
    """
    if isinstance(ell_fn, str):
        key = ell_fn.upper()
        if key not in _ELL_FN_REGISTRY:
            raise ValueError(
                f"Unknown model {ell_fn!r}. Available: {sorted(_ELL_FN_REGISTRY)}"
            )
        ell_fn = _ELL_FN_REGISTRY[key]

    if isinstance(reg_fn, str):
        key = reg_fn.upper()
        if key not in _REG_FN_REGISTRY:
            raise ValueError(
                f"Unknown regularizer {reg_fn!r}. Available: {sorted(_REG_FN_REGISTRY)}"
            )
        _reg_fn, _log_Z_fn, _closed_form_gamma = _REG_FN_REGISTRY[key]
    else:
        _reg_fn             = reg_fn
        _log_Z_fn           = log_Z_ridge
        _closed_form_gamma  = False

    return SimpleNamespace(
        ell_fn               = ell_fn,
        reg_fn               = _reg_fn,
        log_Z_fn             = _log_Z_fn,
        loss_fun             = partial(_loss_fun,               ell_fn),
        validation_fun       = partial(_validation_fun,         ell_fn),
        J_obj                = partial(_J_obj,         ell_fn, _reg_fn),
        J_obj_partial        = partial(_J_obj_partial, ell_fn, _reg_fn),
        theta_hat            = partial(_theta_hat,              ell_fn, _reg_fn),
        ALO                  = partial(_ALO,                    ell_fn, _reg_fn),
        ALO_marginal         = partial(_ALO_marginal,           ell_fn, _reg_fn),
        Q_marginal           = partial(_Q_marginal,             ell_fn),
        U_obj                = partial(U_obj,       ell_fn=ell_fn, reg_fn=_reg_fn, log_Z_fn=_log_Z_fn),
        U_obj_IS             = partial(U_obj_IS,    ell_fn=ell_fn, reg_fn=_reg_fn),
        optimize_ML          = partial(optimize_ML,  ell_fn=ell_fn, reg_fn=_reg_fn),
        optimize_EM          = partial(optimize_EM,  ell_fn=ell_fn, reg_fn=_reg_fn,
                                       log_Z_fn=_log_Z_fn, closed_form_gamma=_closed_form_gamma),
        optimize_ALO         = partial(optimize_ALO, ell_fn=ell_fn, reg_fn=_reg_fn),
        v_theta_fun          = partial(v_theta_fun,  ell_fn=ell_fn, reg_fn=_reg_fn),
        LOO                  = partial(LOO,           ell_fn=ell_fn, reg_fn=_reg_fn),
        LOO_marginal         = partial(LOO_marginal,  ell_fn=ell_fn, reg_fn=_reg_fn),
        LOO_explicit         = partial(LOO_explicit,  ell_fn=ell_fn, reg_fn=_reg_fn),
        GALO                 = partial(GALO,          ell_fn=ell_fn, reg_fn=_reg_fn),
        marginal_uncertainty = partial(marginal_uncertainty, ell_fn=ell_fn, reg_fn=_reg_fn,
                                       log_Z_fn=_log_Z_fn),
        predict_proba        = partial(_predict_proba_generic, ell_fn),
    )




# ═══════════════════════════════════════════════════════════════════════════════
# Legacy: SG_ranking, real_time_ranking
# (Online / stochastic-gradient methods — not used in current paper pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

def SG_ranking(data, params, theta_init=None):
    """Stochastic-gradient online ranking (legacy).  Requires FIVB_paper_figures."""
    if not _FIVB_AVAILABLE:
        raise ImportError("SG_ranking requires FIVB_paper_figures (not available).")

    X, y, u = data
    M, T = X.shape

    if theta_init is None:
        theta = jnp.zeros(M)
    elif isinstance(theta_init, (float, int)):
        theta = jnp.ones(M) * theta_init
    else:
        theta = theta_init.copy()

    theta_all = np.zeros((M, T))
    z_out     = np.zeros(T)

    scale = 1.0 if "scale" not in params else params["scale"]
    eta   = 0.0 if "eta"   not in params else params["eta"]

    xi  = params["weight"][u["category"]]
    hfa = eta * u["hfa"]
    update_step = params["update_step"]

    for t in range(T):
        x_t = X[:, t]
        z_t = jnp.dot(x_t, theta)
        z_out[t] = z_t
        y_t = y[t]

        grad_t = grad(loss_fun)(z_t, y_t, xi[t], hfa[t], params)

        c_FIVB         = params["Ac"] @ params["c"]
        r_FIVB         = params["Ar"] @ params["r"]
        Qy             = _FIVB.FIVB_calculation.Qy_CL(z_t / scale, c_FIVB)
        Expected_score = Qy @ r_FIVB
        grad_FIVB      = Expected_score - r_FIVB[y_t]
        delta_         = grad_FIVB * update_step * xi[t]

        theta     = theta - update_step * (scale ** 2) * x_t * grad_t
        theta_all[:, t] = theta

    return theta_all, z_out


def real_time_ranking(data, params, theta_init=None, v0=None):
    """Real-time (KF / vSKF / SG) ranking (legacy)."""
    X, y, u = data
    M, T = X.shape

    if theta_init is None:
        theta = jnp.zeros(M)
    elif isinstance(theta_init, (float, int)):
        theta = jnp.ones(M) * theta_init
    else:
        theta = theta_init.copy()

    theta_all = np.zeros((M, T))
    z_out     = np.zeros(T)

    xi  = params["weight"][u["category"]]
    hfa_ind = u["hfa"]

    scale            = 1.0 if "scale"            not in params else params["scale"]
    eta              = 0.0 if "eta"              not in params else params["eta"]
    rating_algorithm = params["rating_algorithm"]

    delta_time = jnp.zeros(T)

    if rating_algorithm in {"KF", "vSKF"}:
        beta        = 1.0 if "beta" not in params else params["beta"]
        epsilon     = params["epsilon"]
        delta_time  = jnp.concatenate([jnp.array([0.0]), jnp.diff(u["time_stamp"])])
        if isinstance(v0, (float, int)):
            V_t = (jnp.eye(M) * v0 if rating_algorithm == "KF"
                   else np.ones(M) * v0)

    if rating_algorithm == "SG":
        update_step = params["update_step"]

    V_out = []
    params_wo_string = {k: v for k, v in params.items() if not isinstance(v, str)}
    beta_t = 1.0

    for t in range(T):
        if rating_algorithm in {"KF", "vSKF"}:
            beta_t    = beta ** delta_time[t]
            epsilon_t = epsilon * delta_time[t]

        x_t      = X[:, t]
        z_out[t] = jnp.dot(x_t, theta)
        z_t      = beta_t * jnp.dot(x_t, theta)
        y_t      = y[t]

        grad_t = grad(loss_fun)(z_t / scale + eta * hfa_ind[t], y_t, xi[t], params_wo_string)
        hess_t = hessian(loss_fun)(z_t / scale + eta * hfa_ind[t], y_t, xi[t], params_wo_string)

        if rating_algorithm == "KF":
            V_bar = (beta_t ** 2) * V_t + epsilon_t * jnp.identity(M)
            Vx_t  = V_bar @ x_t
            omega_t = jnp.dot(x_t, Vx_t)
            V_t   = V_bar - jnp.outer(Vx_t, Vx_t * (hess_t / (scale ** 2 + hess_t * omega_t)))
            theta = beta_t * theta - Vx_t * (scale * grad_t) / (scale ** 2 + hess_t * omega_t)
            V_out.append(V_t.copy())

        elif rating_algorithm == "vSKF":
            V_bar   = (beta_t ** 2) * V_t + epsilon_t
            Vx_t    = V_bar * x_t
            omega_t = jnp.dot(x_t, Vx_t)
            V_t     = V_bar * (1 - V_bar * jnp.abs(x_t) * (hess_t / (scale ** 2 + hess_t * omega_t)))
            theta   = beta_t * theta - Vx_t * (scale * grad_t) / (scale ** 2 + hess_t * omega_t)
            V_out.append(V_t.copy())

        elif rating_algorithm == "SG":
            theta = theta - update_step * x_t * grad_t / scale

        theta_all[:, t] = theta

    return theta_all, z_out, V_out
