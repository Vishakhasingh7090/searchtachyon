#!/usr/bin/env python3
"""tachyon_search.py -- single-file version of the analysis code for

    A population-level matched-filter search for tachyonic arrival-time
    advances in gamma-ray bursts and their gravitational-wave counterparts
    (V. Singh, NIT Delhi)

Contents (in order): cosmology, catalog simulation, GLS estimator,
gradient-boosted classifier, all figure/number generation, data export.

Usage
    python tachyon_search.py            # full Monte Carlo (~4 min): figures/, results.json, numbers.tex
    python tachyon_search.py --quick    # reduced Monte Carlo (~40 s)
    python tachyon_search.py --export   # also write data/*.csv after the run

Requires numpy, scipy, matplotlib, scikit-learn.
"""
from __future__ import annotations

import argparse, csv, json, os, time
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc



# ============================================================================
# 1. Cosmology: dispersion, velocity excess, time-of-flight kernels
# ============================================================================

H0_KM_S_MPC = 67.7
OMEGA_M = 0.31
OMEGA_L = 0.69
MPC_IN_KM = 3.0856776e19
H0_INV_S = MPC_IN_KM / H0_KM_S_MPC          # Hubble time in seconds
EV_TO_KG = 1.782662e-36                       # 1 eV/c^2 in kg


def hubble_h(z):
    """Dimensionless Hubble rate h(z) = H(z)/H0."""
    z = np.asarray(z, dtype=float)
    return np.sqrt(OMEGA_M * (1.0 + z) ** 3 + OMEGA_L)


def _kernel(z, power):
    """I_n(z) = int_0^z dz' / [(1+z')^n h(z')], scalar or array z."""
    def one(zz):
        if zz <= 0:
            return 0.0
        val, _ = quad(lambda x: 1.0 / ((1.0 + x) ** power * hubble_h(x)), 0.0, zz)
        return val
    return np.vectorize(one, otypes=[float])(z)


def kernel_I1(z):
    """Single-power ("naive") kernel int dz/[(1+z)h]."""
    return _kernel(z, 1)


def kernel_I2(z):
    """Inverse-square kernel int dz/[(1+z)^2 h] appropriate to E^-2 dispersion."""
    return _kernel(z, 2)


def group_velocity_excess(E, E_t):
    """v/c - 1 = sqrt(1 + (E_t/E)^2) - 1 for a tachyon of observed energy E."""
    E = np.asarray(E, dtype=float)
    return np.sqrt(1.0 + (E_t / E) ** 2) - 1.0


def velocity_excess_along_path(z, eps):
    """Exact local velocity excess delta(z) for epsilon = E_t^2/(E_obs^2+E_t^2).

    The physical momentum redshifts as p(z) = p_obs (1+z), so the velocity excess
    is smaller in the past and grows as the quantum ages.
    """
    z = np.asarray(z, dtype=float)
    # (1-x)^(-1/2) - 1 evaluated without cancellation for x ~ 1e-18
    x = eps / (1.0 + z) ** 2
    return np.expm1(-0.5 * np.log1p(-x))


def arrival_advance_exact(E_obs, E_t, z):
    """Exact arrival advance Delta t = H0^-1 int_0^z delta(z')/h(z') dz'  [s]."""
    eps = E_t ** 2 / (E_obs ** 2 + E_t ** 2)

    def one(zz):
        if zz <= 0:
            return 0.0
        val, _ = quad(lambda x: velocity_excess_along_path(x, eps) / hubble_h(x), 0.0, zz)
        return H0_INV_S * val
    return np.vectorize(one, otypes=[float])(z)


def arrival_advance_lo(E_obs, E_t, z):
    """Leading-order advance 0.5 (E_t/E)^2 H0^-1 I2(z)  [s]  (Eq. 3 of the paper)."""
    return 0.5 * (E_t / np.asarray(E_obs, float)) ** 2 * H0_INV_S * kernel_I2(z)


def arrival_advance_naive(E_obs, E_t, z):
    """Same but with the single-power kernel I1 (for comparison only)."""
    return 0.5 * (E_t / np.asarray(E_obs, float)) ** 2 * H0_INV_S * kernel_I1(z)


def arrival_advance_flat(E_obs, E_t, z):
    """Low-redshift flat-space limit (E_t/E)^2 L/(2c) with L = cz/H0."""
    return 0.5 * (E_t / np.asarray(E_obs, float)) ** 2 * H0_INV_S * np.asarray(z, float)


def template_K(E_b, E_ref, z, I2=None):
    """Matched-filter template K_b (Eq. 5): Delta t_b - Delta t_ref per unit theta=E_t^2.

    E_ref = np.inf encodes an anchored event whose reference is a luminal (GW) clock.
    Returns seconds per eV^2.
    """
    if I2 is None:
        I2 = kernel_I2(z)
    E_b = np.asarray(E_b, float)
    E_ref = np.asarray(E_ref, float)
    return (1.0 / E_b ** 2 - 1.0 / E_ref ** 2) * 0.5 * H0_INV_S * I2


def tachyon_mass_kg(E_t_eV):
    """|mu| = E_t/c^2 in kg."""
    return E_t_eV * EV_TO_KG


# ============================================================================
# 2. Synthetic multi-messenger catalogs
# ============================================================================

BANDS = ("MeV", "GeV", "TeV", "nu")
BAND_ENERGY = {"MeV": 5e5, "GeV": 1e9, "TeV": 1e12, "nu": 1e14}
BAND_TIMING = {"MeV": 0.010, "GeV": 0.100, "TeV": 0.500, "nu": 1.000}
BAND_PROB = {"MeV": 1.00, "GeV": 0.45, "TeV": 0.15, "nu": 0.08}


@dataclass
class SimConfig:
    n_events: int = 400
    sigma_lag: float = 0.50          # intrinsic spectral-lag scatter per band [s]
    mean_lag_per_decade: float = 0.0  # systematic lag [s] per decade of log10(E_b/E_ref); + means high-E trails
    lag_df: float = np.inf           # Student-t dof for lags (inf = Gaussian)
    f_anchored: float = 0.20         # fraction with GW trigger
    jet_delay_mean: float = 0.0      # mean positive jet-launch delay for anchored events [s]
    jet_delay_sigma: float = 0.0
    z_median: float = 0.8
    z_sigma_ln: float = 0.7
    z_max: float = 5.0
    z_err_frac: float = 0.0          # sigma_z/(1+z) applied to the *measured* redshift
    liv_linear_coeff: float = 0.0    # sub-luminal linear LIV: t_b += +c1 * E_b/eV * H0^-1 int (1+z)/h  (c1 in eV^-1 units of 1/E_QG)
    E_t_inject_meV: float = 0.0
    seed: int = 0


@dataclass
class Catalog:
    """Flattened catalog: one row per (event, non-reference band)."""
    event: np.ndarray        # event index
    band: np.ndarray         # band index into BANDS
    E: np.ndarray            # band energy [eV]
    E_ref: np.ndarray        # reference energy [eV] (inf if anchored)
    z_true: np.ndarray
    z_meas: np.ndarray
    anchored: np.ndarray     # bool
    t: np.ndarray            # observed band time relative to reference [s]
    K: np.ndarray            # template using measured z [s/eV^2]
    sigma_diag: np.ndarray   # independent variance term per row
    sigma_shared: np.ndarray # shared variance term per row (reference band), 0 if anchored
    n_events: int = 0
    config: SimConfig = field(default_factory=SimConfig)

    def per_event_blocks(self):
        """Yield (rows, K_i, t_i, Sigma_i) for each event."""
        order = np.argsort(self.event, kind="stable")
        ev = self.event[order]
        starts = np.r_[0, np.flatnonzero(np.diff(ev)) + 1, len(ev)]
        for a, b in zip(starts[:-1], starts[1:]):
            rows = order[a:b]
            n = len(rows)
            S = np.diag(self.sigma_diag[rows]) + self.sigma_shared[rows][0] * np.ones((n, n))
            yield rows, self.K[rows], self.t[rows], S


def _draw_redshifts(rng, n, cfg: SimConfig):
    z = rng.lognormal(mean=np.log(cfg.z_median), sigma=cfg.z_sigma_ln, size=n)
    return np.clip(z, 0.005, cfg.z_max)


def _draw_lags(rng, n, cfg: SimConfig):
    if np.isfinite(cfg.lag_df):
        # Student-t scaled so that the *core* width matches sigma_lag
        return cfg.sigma_lag * rng.standard_t(cfg.lag_df, size=n)
    return rng.normal(0.0, cfg.sigma_lag, size=n)


def simulate_catalog(cfg: SimConfig, rng: np.random.Generator | None = None) -> Catalog:
    rng = rng if rng is not None else np.random.default_rng(cfg.seed)
    n = cfg.n_events
    theta = (cfg.E_t_inject_meV * 1e-3) ** 2  # eV^2

    z_true = _draw_redshifts(rng, n, cfg)
    if cfg.z_err_frac > 0:
        z_meas = z_true + rng.normal(0, cfg.z_err_frac * (1 + z_true))
        z_meas = np.clip(z_meas, 0.005, cfg.z_max)
    else:
        z_meas = z_true.copy()
    I2_true = kernel_I2(z_true)
    I2_meas = kernel_I2(z_meas)
    anchored = rng.random(n) < cfg.f_anchored

    # which bands are present
    present = np.column_stack([rng.random(n) < BAND_PROB[b] for b in BANDS])
    present[:, 0] = True

    # intrinsic emission-time offsets per (event, band) and jet delay
    lag = _draw_lags(rng, n * len(BANDS), cfg).reshape(n, len(BANDS))
    if cfg.mean_lag_per_decade != 0.0:
        # systematic component: high-energy photons trail by mean_lag * log10(E_b/E_MeV)
        logE = np.log10([BAND_ENERGY[b] / BAND_ENERGY["MeV"] for b in BANDS])
        lag += cfg.mean_lag_per_decade * logE[None, :]
    jet = np.zeros(n)
    if anchored.any() and (cfg.jet_delay_mean > 0 or cfg.jet_delay_sigma > 0):
        jet[anchored] = np.abs(rng.normal(cfg.jet_delay_mean, cfg.jet_delay_sigma, anchored.sum()))

    # optional sub-luminal linear LIV delay (Jacob & Piran n=1 kernel)
    if cfg.liv_linear_coeff != 0.0:
        Ilin = np.vectorize(lambda zz: quad(lambda x: (1 + x) / hubble_h(x), 0, zz)[0])(z_true)
    else:
        Ilin = np.zeros(n)

    rows = []
    for i in range(n):
        bands_i = [j for j in range(len(BANDS)) if present[i, j]]
        if anchored[i]:
            ref_j = None
            E_ref = np.inf
        else:
            ref_j = bands_i[-1]
            E_ref = BAND_ENERGY[BANDS[ref_j]]
            bands_i = bands_i[:-1]
            if not bands_i:      # single-band unanchored event carries no information
                continue
        ref_noise = 0.0 if ref_j is None else rng.normal(0, BAND_TIMING[BANDS[ref_j]])
        for j in bands_i:
            b = BANDS[j]
            E_b = BAND_ENERGY[b]
            # true propagation term
            K_true = template_K(E_b, E_ref, z_true[i], I2=I2_true[i])
            liv = 0.0
            if cfg.liv_linear_coeff != 0.0:
                E_r = 0.0 if ref_j is None else BAND_ENERGY[BANDS[ref_j]]
                liv = cfg.liv_linear_coeff * (E_b - E_r) * H0_INV_S * Ilin[i]
            t_int = lag[i, j] + jet[i] - (0.0 if ref_j is None else lag[i, ref_j])
            noise = rng.normal(0, BAND_TIMING[b]) - ref_noise
            t_obs = -theta * K_true + liv + t_int + noise
            sig_diag = cfg.sigma_lag ** 2 + BAND_TIMING[b] ** 2
            sig_shared = (cfg.jet_delay_sigma ** 2 if ref_j is None
                          else cfg.sigma_lag ** 2 + BAND_TIMING[BANDS[ref_j]] ** 2)
            rows.append((i, j, E_b, E_ref, z_true[i], z_meas[i], anchored[i], t_obs,
                         template_K(E_b, E_ref, z_meas[i], I2=I2_meas[i]), sig_diag, sig_shared))

    arr = np.array(rows, dtype=object)
    cat = Catalog(
        event=arr[:, 0].astype(int), band=arr[:, 1].astype(int), E=arr[:, 2].astype(float),
        E_ref=arr[:, 3].astype(float), z_true=arr[:, 4].astype(float), z_meas=arr[:, 5].astype(float),
        anchored=arr[:, 6].astype(bool), t=arr[:, 7].astype(float), K=arr[:, 8].astype(float),
        sigma_diag=arr[:, 9].astype(float), sigma_shared=arr[:, 10].astype(float),
        n_events=n, config=cfg,
    )
    return cat


# ============================================================================
# 3. GLS matched-filter estimator, limits, nuisance templates, Huber, bootstrap
# ============================================================================

Z95 = 1.6448536269514722


def _design(cat: Catalog, nuisance: tuple[str, ...] = ()):
    """Return list of (rows, design matrix A_i, t_i, Sigma_i).

    Column 0 of A_i is -K_i (theta); subsequent columns are nuisance templates.
    """
    blocks = []
    Ilin_cache = {}
    for rows, K, t, S in cat.per_event_blocks():
        cols = [-K]
        for name in nuisance:
            if name == "meanlag":
                # log10(E_b/E_ref) with E_ref -> log10(E_b/E_MeV) for anchored (absolute lag)
                Eref = cat.E_ref[rows]
                base = np.where(np.isfinite(Eref), Eref, BAND_ENERGY["MeV"])
                cols.append(np.log10(cat.E[rows] / base))
            elif name == "liv1":
                z = cat.z_meas[rows][0]
                if z not in Ilin_cache:
                    Ilin_cache[z] = quad(lambda x: (1 + x) / hubble_h(x), 0, z)[0]
                Eref = cat.E_ref[rows]
                Er = np.where(np.isfinite(Eref), Eref, 0.0)
                cols.append((cat.E[rows] - Er) * H0_INV_S * Ilin_cache[z])
            elif name == "jet":
                cols.append(cat.anchored[rows].astype(float))
            else:
                raise ValueError(name)
        A = np.column_stack(cols)
        blocks.append((rows, A, t, S))
    return blocks


def gls_fit(cat: Catalog, nuisance: tuple[str, ...] = (), weights: np.ndarray | None = None):
    """Generalised least squares.  Returns (beta_hat, cov, chi2, ndof).

    beta_hat[0] is theta_hat [eV^2].  `weights` (per row) down-weight rows for
    robust IRLS; they multiply the inverse covariance diagonally.
    """
    blocks = _design(cat, nuisance)
    p = blocks[0][1].shape[1]
    ATA = np.zeros((p, p))
    ATt = np.zeros(p)
    chi2 = 0.0
    n = 0
    for rows, A, t, S in blocks:
        Sinv = np.linalg.inv(S)
        if weights is not None:
            w = np.sqrt(weights[rows])
            Sinv = w[:, None] * Sinv * w[None, :]
        ATA += A.T @ Sinv @ A
        ATt += A.T @ Sinv @ t
        n += len(rows)
    cov = np.linalg.inv(ATA)
    beta = cov @ ATt
    for rows, A, t, S in blocks:
        r = t - A @ beta
        chi2 += r @ np.linalg.solve(S, r)
    return beta, cov, chi2, n - p


def upper_limit(theta_hat, sigma_theta, cl=0.95):
    """One-sided frequentist limit on E_t [eV] with the physical boundary theta >= 0."""
    zc = norm.ppf(cl)
    theta_ul = max(theta_hat, 0.0) + zc * sigma_theta
    return np.sqrt(theta_ul)


def bayesian_upper_limit(theta_hat, sigma_theta, cl=0.95):
    """Limit from the Gaussian likelihood truncated to theta >= 0 (flat prior on theta)."""
    # posterior ∝ N(theta; theta_hat, sigma) for theta>=0
    a = norm.cdf(-theta_hat / sigma_theta)          # mass below zero
    q = a + cl * (1 - a)
    theta_ul = theta_hat + sigma_theta * norm.ppf(q)
    return np.sqrt(max(theta_ul, 0.0))


def profile_likelihood(theta_grid, theta_hat, sigma_theta):
    """-ln(L/Lmax) on a grid (quadratic for a linear Gaussian model)."""
    return 0.5 * ((theta_grid - theta_hat) / sigma_theta) ** 2


def robust_fit(cat: Catalog, nuisance=(), c=1.345, n_iter=20):
    """Huber-weighted iteratively reweighted GLS for heavy-tailed residuals."""
    weights = np.ones(len(cat.t))
    beta = cov = None
    for _ in range(n_iter):
        beta, cov, _, _ = gls_fit(cat, nuisance, weights)
        blocks = _design(cat, nuisance)
        new_w = np.ones_like(weights)
        for rows, A, t, S in blocks:
            r = (t - A @ beta) / np.sqrt(np.diag(S))
            a = np.abs(r)
            new_w[rows] = np.where(a <= c, 1.0, c / np.maximum(a, 1e-12))
        if np.allclose(new_w, weights, atol=1e-4):
            weights = new_w
            break
        weights = new_w
    # inflate covariance for the efficiency loss of the Huber weights
    return beta, cov, weights


def information_budget(cat: Catalog):
    """Fraction of the Fisher information on theta contributed by each (band, anchoring) class."""
    info = {}
    for rows, K, t, S in cat.per_event_blocks():
        Sinv = np.linalg.inv(S)
        # per-row contribution: diag of K Sinv K (approximate attribution)
        contrib = K * (Sinv @ K)
        for r, c in zip(rows, contrib):
            key = (BANDS[cat.band[r]], "anchored" if cat.anchored[r] else "unanchored")
            info[key] = info.get(key, 0.0) + c
    tot = sum(info.values())
    return {f"{b}/{a}": v / tot for (b, a), v in sorted(info.items())}


def bootstrap_theta(cat: Catalog, n_boot=500, rng=None):
    """Event-level bootstrap of theta_hat."""
    rng = rng if rng is not None else np.random.default_rng(1)
    blocks = _design(cat)
    # precompute per-event sufficient statistics
    num = np.array([(A.T @ np.linalg.solve(S, t))[0] for _, A, t, S in blocks])
    den = np.array([(A.T @ np.linalg.solve(S, A))[0, 0] for _, A, t, S in blocks])
    m = len(num)
    out = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, m, m)
        out[k] = num[idx].sum() / den[idx].sum()
    return out


# ============================================================================
# 4. Gradient-boosted catalog-level classifier
# ============================================================================

FEATURE_NAMES = []
for _b in BANDS:
    for _a in ("anch", "unanch"):
        for _f in ("corr", "slope", "mean", "std", "n"):
            FEATURE_NAMES.append(f"{_b}_{_a}_{_f}")


def catalog_features(cat: Catalog) -> np.ndarray:
    feats = []
    for j, b in enumerate(BANDS):
        for anch in (True, False):
            m = (cat.band == j) & (cat.anchored == anch)
            n = int(m.sum())
            if n >= 3:
                K, t = cat.K[m], cat.t[m]
                w = 1.0 / (cat.sigma_diag[m] + cat.sigma_shared[m])
                corr = np.corrcoef(K, t)[0, 1] if K.std() > 0 else 0.0
                slope = np.sum(w * K * t) / np.sum(w * K * K)
                feats += [corr, slope, t.mean(), t.std(), n]
            else:
                feats += [0.0, 0.0, 0.0, 0.0, n]
    return np.array(feats)


def build_training_set(E_t_meV, n_per_class=150, n_events=120, base_cfg: SimConfig | None = None, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    base = base_cfg or SimConfig()
    for label, Et in ((0, 0.0), (1, E_t_meV)):
        for k in range(n_per_class):
            cfg = SimConfig(**{**base.__dict__, "n_events": n_events, "E_t_inject_meV": Et,
                               "seed": int(rng.integers(0, 2**31))})
            X.append(catalog_features(simulate_catalog(cfg)))
            y.append(label)
    return np.array(X), np.array(y)


def cross_validated_roc(X, y, n_splits=5, seed=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = np.zeros(len(y))
    importances = np.zeros(X.shape[1])
    for tr, te in skf.split(X, y):
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                             l2_regularization=1.0, random_state=seed)
        clf.fit(X[tr], y[tr])
        scores[te] = clf.predict_proba(X[te])[:, 1]
    fpr, tpr, _ = roc_curve(y, scores)
    return fpr, tpr, auc(fpr, tpr), scores


# ============================================================================
# 5. Figures, results.json, numbers.tex
# ============================================================================




plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "figure.dpi": 130, "savefig.bbox": "tight", "lines.linewidth": 1.4,
                     "font.family": "serif", "mathtext.fontset": "cm"})
FIG = "figures"
os.makedirs(FIG, exist_ok=True)
R = {}          # results dictionary -> results.json / numbers.tex
T0 = time.time()


def savefig(fig, name):
    """Write figures/<name>.png (300 dpi, used by the paper) and figures/<name>.pdf."""
    fig.savefig(f"{FIG}/{name}.png", dpi=300)
    fig.savefig(f"{FIG}/{name}.pdf")
    plt.close(fig)


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def ul_from_cat(cat, **kw):
    b, c, chi2, nd = gls_fit(cat, **kw)
    return b[0], np.sqrt(c[0, 0]), chi2 / nd


# --------------------------------------------------------------------------- #
# 1. Kinematics and kernels (Figs 1-5)
# --------------------------------------------------------------------------- #
def fig_kinematics():
    E = np.logspace(5, 15, 300)
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for Et, col in ((1e3, "C0"), (1e5, "C3"), (1e7, "C2")):
        ax[0].loglog(E, group_velocity_excess(E, Et), color=col, label=rf"$E_t=10^{{{int(np.log10(Et))}}}$ eV")
        ax[1].loglog(E, arrival_advance_lo(E, Et, 1.0), color=col, label=rf"$E_t=10^{{{int(np.log10(Et))}}}$ eV")
    ax[1].axhspan(1e-3, 1, color="gray", alpha=0.2, lw=0)
    ax[1].text(2e13, 3e-2, "sub-second\ntiming reach", fontsize=7, color="gray", ha="center")
    ax[0].set(xlabel="observed energy $E$ [eV]", ylabel="$v/c-1$", title="(a) superluminal excess")
    ax[1].set(xlabel="observed energy $E$ [eV]", ylabel=r"arrival advance $\Delta t$ [s] ($z=1$)", title="(b) time-of-flight advance")
    for a in ax:
        a.legend(); a.grid(alpha=0.3, which="major")
    savefig(fig, "fig1_kinematics")


def fig_velocity_profile():
    z = np.linspace(0, 4, 200)
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ax[0].semilogy(z, (1 + z) ** -2, label=r"$\delta\propto(1+z)^{-2}$ (exact LO)")
    ax[0].semilogy(z, (1 + z) ** -1, "--", color="C3", label=r"$(1+z)^{-1}$ (naive)")
    ax[0].set(xlabel="redshift $z$", ylabel=r"velocity excess $\delta(z)/\delta(0)$", title=r"(a) physical regime ($\varepsilon\to0$)")
    for eps in (0.1, 0.5, 0.9, 0.99):
        ax[1].semilogy(z, velocity_excess_along_path(z, eps), label=rf"$\varepsilon={eps}$")
    ax[1].set(xlabel="redshift $z$", ylabel=r"velocity excess $\delta(z)=v/c-1$", title="(b) exact, strong regime")
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    savefig(fig, "fig2_velocity_profile")


def fig_kernels():
    z = np.linspace(0, 3, 120)
    I1, I2 = kernel_I1(z), kernel_I2(z)
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ax[0].plot(z, I1, "--", color="C3", label=r"$I_1=\int dz/[(1+z)h]$ (naive)")
    ax[0].plot(z, I2, color="C0", label=r"$I_2=\int dz/[(1+z)^2 h]$ (this work)")
    ax[0].set(xlabel="redshift $z$", ylabel="kernel value", title="(a) cosmological kernel")
    with np.errstate(invalid="ignore", divide="ignore"):
        r = I1 / I2
    ax[1].plot(z[1:], r[1:], color="C2", label=r"$I_1/I_2$ (advance bias)")
    ax[1].plot(z[1:], np.sqrt(r[1:]), "-.", color="C4", label=r"$\sqrt{I_1/I_2}$ ($E_t$-limit bias)")
    ax[1].set(xlabel="redshift $z$", ylabel="ratio", title="(b) error of the naive kernel")
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    savefig(fig, "fig3_kernels")
    R["I2_z1"] = float(kernel_I2(1.0)); R["I1_z1"] = float(kernel_I1(1.0))
    R["kernel_bias_z0p1"] = float(kernel_I1(0.1) / kernel_I2(0.1) - 1) * 100
    R["kernel_bias_z3"] = float(kernel_I1(3.0) / kernel_I2(3.0) - 1) * 100
    R["Et_bias_z3"] = float(np.sqrt(kernel_I1(3.0) / kernel_I2(3.0)) - 1) * 100
    # low-z check of the flat-space limit
    R["flat_limit_err_z0p01"] = float(abs(arrival_advance_flat(1e6, 1e-3, 0.01) / arrival_advance_lo(1e6, 1e-3, 0.01) - 1) * 100)


def fig_advance_vs_z():
    z = np.linspace(0, 3, 60)
    E, Et = 1e6, 1e-3
    naive = arrival_advance_naive(E, Et, z) * 1e3
    lo = arrival_advance_lo(E, Et, z) * 1e3
    ex = arrival_advance_exact(E, Et, z) * 1e3
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(z, naive, "--", color="C3", label="naive LO (single power)")
    ax.plot(z, lo, color="C0", label="corrected LO (this work)")
    ax.plot(z, ex, ":", color="C2", lw=2.2, label="exact")
    ax.set(xlabel="source redshift $z$", ylabel=r"arrival advance $\Delta t$ [ms]", title=r"$E_{\rm obs}=1$ MeV, $E_t=1$ meV")
    ax.legend(); ax.grid(alpha=0.3)
    savefig(fig, "fig4_advance_vs_z")
    R["dt_1MeV_1meV_z1_ms"] = float(arrival_advance_lo(E, Et, 1.0) * 1e3)
    R["exact_vs_lo_maxdev_pct"] = float(np.max(np.abs(ex[1:] / lo[1:] - 1)) * 100)


def fig_bands_vs_z():
    z = np.linspace(0.01, 3, 60)
    Et = 1e-3
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    labels = {"MeV": "0.5 MeV (GBM)", "GeV": "1 GeV (LAT)", "TeV": "1 TeV (IACT)", "nu": r"100 TeV $\nu$"}
    for b in BANDS:
        ax.semilogy(z, arrival_advance_lo(BAND_ENERGY[b], Et, z), label=labels[b])
    ax.set(xlabel="redshift $z$", ylabel=r"arrival advance $\Delta t$ [s]", title=r"$E_t=1$ meV, by band")
    ax.legend(); ax.grid(alpha=0.3)
    savefig(fig, "fig5_bands_vs_z")


# --------------------------------------------------------------------------- #
# 2. Estimator validation on the fiducial catalog (Figs 6, 8)
# --------------------------------------------------------------------------- #
def fiducial_and_profile(seed=42):
    cfg = SimConfig(n_events=400, seed=seed)
    cat = simulate_catalog(cfg)
    th, sg, chi2r = ul_from_cat(cat)
    ul = upper_limit(th, sg); ulb = bayesian_upper_limit(th, sg)
    R.update(fid_theta_eV2=float(th), fid_sigma_eV2=float(sg), fid_UL_meV=float(ul * 1e3),
             fid_UL_bayes_meV=float(ulb * 1e3), fid_chi2_red=float(chi2r), fid_rows=int(len(cat.t)),
             fid_n_anchored=int(cat.anchored[np.unique(cat.event, return_index=True)[1]].sum()),
             fid_theta_sig=float(th / sg), fid_z_median=float(np.median(cat.z_true)))
    grid = np.linspace(th - 4 * sg, th + 4 * sg, 400)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(grid * 1e6, profile_likelihood(grid, th, sg))
    ax.axvline(ul ** 2 * 1e6, ls="--", color="C3", label=rf"95% UL: $E_t<{ul*1e3:.2f}$ meV")
    ax.axhline(0.5 * norm.ppf(0.95) ** 2, color="gray", lw=0.8)
    ax.axvline(0, color="k", lw=0.6)
    ax.set(xlabel=r"$\theta=E_t^2$ [$10^{-6}$ eV$^2$]", ylabel=r"$-\ln\mathcal{L}/\mathcal{L}_{\max}$", title="profile likelihood (null catalog)")
    ax.legend(); ax.grid(alpha=0.3)
    savefig(fig, "fig8_profile_likelihood")

    # information budget and bootstrap
    R["info_budget"] = {k: float(v) for k, v in information_budget(cat).items()}
    boot = bootstrap_theta(cat, n_boot=1000)
    R["boot_sigma_over_analytic"] = float(boot.std() / sg)
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ax[0].hist(boot * 1e6, bins=40, density=True, alpha=0.6, label="event bootstrap (1000)")
    xx = np.linspace(boot.min(), boot.max(), 200)
    ax[0].plot(xx * 1e6, norm.pdf(xx, th, sg), color="C3", label=rf"analytic $\mathcal{{N}}(\hat\theta,\sigma_{{\hat\theta}})$, ratio {boot.std()/sg:.2f}")
    ax[0].set(xlabel=r"$\hat\theta$ [$10^{-6}$ eV$^2$]", ylabel="density", title="(a) bootstrap check of the error")
    ax[0].legend()
    ev_info, ev_z, ev_anch = [], [], []
    for rows, K, t, S in cat.per_event_blocks():
        ev_info.append(K @ np.linalg.solve(S, K)); ev_z.append(cat.z_true[rows][0]); ev_anch.append(cat.anchored[rows][0])
    ev_info, ev_z, ev_anch = np.array(ev_info), np.array(ev_z), np.array(ev_anch)
    tot = ev_info.sum()
    R["info_frac_anchored_events"] = float(ev_info[ev_anch].sum() / tot)
    R["info_frac_z_above_1"] = float(ev_info[ev_z > 1].sum() / tot)
    R["frac_events_z_above_1"] = float(np.mean(ev_z > 1))
    ax[1].scatter(ev_z[~ev_anch], ev_info[~ev_anch] / tot * 100, s=8, alpha=0.6, label="unanchored")
    ax[1].scatter(ev_z[ev_anch], ev_info[ev_anch] / tot * 100, s=8, alpha=0.8, color="C3", label="GW-anchored")
    ax[1].set(xlabel="redshift $z$", ylabel="share of Fisher information [%]", title="(b) which events carry the leverage", yscale="log")
    ax[1].legend(loc="lower right")
    savefig(fig, "fig12_bootstrap_infobudget")
    return cat


def fig_linearity(n_cat=100, n_events=400):
    inj = np.linspace(0, 3, 7)  # meV
    rec_mean, rec_std, theta_mean = [], [], []
    for Et in inj:
        vals, ths = [], []
        for k in range(n_cat):
            cat = simulate_catalog(SimConfig(n_events=n_events, E_t_inject_meV=Et, seed=1500 + k))
            th, sg, _ = ul_from_cat(cat)
            vals.append(np.sqrt(max(th, 0)) * 1e3); ths.append(th)
        rec_mean.append(np.mean(vals)); rec_std.append(np.std(vals)); theta_mean.append(np.mean(ths))
    rec_mean, rec_std = np.array(rec_mean), np.array(rec_std)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot([0, 3], [0, 3], "--", color="gray", label="ideal")
    ax.errorbar(inj, rec_mean, rec_std, fmt="o", ms=4, capsize=2, label=f"recovered ({n_cat} catalogs)")
    ax.set(xlabel="injected $E_t$ [meV]", ylabel=r"recovered $\hat E_t$ [meV]", title="estimator linearity")
    ax.legend(); ax.grid(alpha=0.3)
    savefig(fig, "fig6_linearity")
    R["linearity_max_abs_bias_meV"] = float(np.max(np.abs(rec_mean[1:] - inj[1:])))
    R["linearity_max_rel_bias_pct"] = float(np.max(np.abs(rec_mean[1:] / inj[1:] - 1)) * 100)
    theta_mean = np.array(theta_mean)
    R["linearity_theta_max_rel_bias_pct"] = float(np.max(np.abs(theta_mean[1:] / (inj[1:] * 1e-3) ** 2 - 1)) * 100)


# --------------------------------------------------------------------------- #
# 3. Classifier (Fig 7)
# --------------------------------------------------------------------------- #
def fig_classifier(scales=(0.3, 0.5, 1.0), n_per_class=150, n_events=120):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    aucs = {}
    for Et, col in zip(scales, ("C0", "C3", "C2")):
        X, y = build_training_set(Et, n_per_class=n_per_class, n_events=n_events, seed=int(Et * 100))
        fpr, tpr, a, _ = cross_validated_roc(X, y)
        aucs[f"{Et:.1f}"] = float(a)
        ax.plot(fpr, tpr, color=col, label=rf"$E_t={Et}$ meV (AUC={a:.2f})")
        log(f"  classifier Et={Et} meV AUC={a:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
    ax.set(xlabel="false-alarm rate", ylabel="detection rate", title="catalog-level discrimination")
    ax.legend(); ax.grid(alpha=0.3)
    savefig(fig, "fig7_roc")
    R["auc"] = aucs
    # matched-filter equivalent at 120 events for comparison
    uls = [upper_limit(*ul_from_cat(simulate_catalog(SimConfig(n_events=n_events, seed=7000 + k)))[:2]) * 1e3 for k in range(30)]
    R["UL_120_median_meV"] = float(np.median(uls))


# --------------------------------------------------------------------------- #
# 4. Sensitivity scaling (Fig 9)
# --------------------------------------------------------------------------- #
def fig_scaling(Ns=(25, 50, 100, 200, 400, 800, 1600), n_real=30):
    med, lo, hi = [], [], []
    for N in Ns:
        uls = [upper_limit(*ul_from_cat(simulate_catalog(SimConfig(n_events=N, seed=2000 + k)))[:2]) * 1e3 for k in range(n_real)]
        med.append(np.median(uls)); lo.append(np.percentile(uls, 16)); hi.append(np.percentile(uls, 84))
    Ns = np.array(Ns); med = np.array(med)
    slope = np.polyfit(np.log(Ns), np.log(med), 1)[0]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.fill_between(Ns, lo, hi, alpha=0.2, label="16-84% scatter")
    ax.loglog(Ns, med, "o-", ms=4, label="median 95% UL")
    ref = med[Ns == 400][0] * (Ns / 400.0) ** -0.25
    ax.loglog(Ns, ref, "--", color="gray", label=r"$\propto N^{-1/4}$")
    ax.set(xlabel="number of events $N$", ylabel="95% upper limit on $E_t$ [meV]", title="sensitivity scaling")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    savefig(fig, "fig9_scaling")
    R["scaling"] = {str(int(n)): float(m) for n, m in zip(Ns, med)}
    R["scaling_fitted_slope"] = float(slope)


# --------------------------------------------------------------------------- #
# 5. New robustness analyses
# --------------------------------------------------------------------------- #
def coverage(Ets=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), n_sim=300, n_events=400):
    cov_f, cov_b, pull_mean, pull_std = [], [], [], []
    for Et in Ets:
        ulf, ulb, pulls = [], [], []
        for k in range(n_sim):
            cat = simulate_catalog(SimConfig(n_events=n_events, E_t_inject_meV=Et, seed=3000 + k + int(Et * 1000)))
            th, sg, _ = ul_from_cat(cat)
            ulf.append(upper_limit(th, sg)); ulb.append(bayesian_upper_limit(th, sg))
            pulls.append((th - (Et * 1e-3) ** 2) / sg)
        ulf, ulb = np.array(ulf) * 1e3, np.array(ulb) * 1e3
        cov_f.append(np.mean(ulf >= Et)); cov_b.append(np.mean(ulb >= Et))
        pull_mean.append(np.mean(pulls)); pull_std.append(np.std(pulls))
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ax[0].plot(Ets, np.array(cov_f) * 100, "o-", label="frequentist (boundary-clipped)")
    ax[0].plot(Ets, np.array(cov_b) * 100, "s--", label=r"flat prior on $\theta\geq0$")
    ax[0].axhline(95, color="gray", lw=0.8)
    ax[0].set(xlabel="injected $E_t$ [meV]", ylabel="fraction of ULs above truth [%]", title="(a) coverage of the 95% limit", ylim=(88, 101))
    ax[0].legend()
    ax[1].errorbar(Ets, pull_mean, pull_std, fmt="o", capsize=3)
    ax[1].axhline(0, color="gray", lw=0.8); ax[1].axhline(1, color="gray", lw=0.5, ls=":"); ax[1].axhline(-1, color="gray", lw=0.5, ls=":")
    ax[1].set(xlabel="injected $E_t$ [meV]", ylabel=r"pull $(\hat\theta-\theta_{\rm true})/\sigma_{\hat\theta}$", title="(b) pull distribution")
    savefig(fig, "fig10_coverage")
    R["coverage"] = {f"{e:.1f}": {"freq": float(f), "bayes": float(b), "pull_mean": float(m), "pull_std": float(s)}
                     for e, f, b, m, s in zip(Ets, cov_f, cov_b, pull_mean, pull_std)}


def mean_lag_bias(lags=(-0.3, -0.15, 0.0, 0.15, 0.3), n_sim=40, n_events=400):
    rows = {"plain": [], "joint": [], "anchored_only": []}
    errs = {k: [] for k in rows}
    for tau in lags:
        vals = {k: [] for k in rows}
        for k in range(n_sim):
            cfg = SimConfig(n_events=n_events, mean_lag_per_decade=tau, seed=4000 + k)
            cat = simulate_catalog(cfg)
            vals["plain"].append(ul_from_cat(cat)[0])
            vals["joint"].append(ul_from_cat(cat, nuisance=("meanlag",))[0])
            # anchored-only sub-catalog
            m = cat.anchored
            sub = Catalog(**{f: (getattr(cat, f)[m] if isinstance(getattr(cat, f), np.ndarray) else getattr(cat, f)) for f in cat.__dataclass_fields__})
            vals["anchored_only"].append(ul_from_cat(sub, nuisance=("meanlag",))[0])
        for k in rows:
            rows[k].append(np.mean(vals[k]) * 1e6); errs[k].append(np.std(vals[k]) / np.sqrt(n_sim) * 1e6)
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for k, lab, mk in (("plain", r"tachyon template only", "o"), ("joint", r"joint fit with $\log_{10}E$ lag template", "s"),
                       ("anchored_only", "anchored events only, joint fit", "^")):
        ax.errorbar(lags, rows[k], errs[k], fmt=mk + "-", ms=4, capsize=2, label=lab)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(xlabel="mean intrinsic lag [s per decade of energy]", ylabel=r"$\langle\hat\theta\rangle$ [$10^{-6}$ eV$^2$]", title="bias from a systematic spectral lag")
    ax.legend(); ax.grid(alpha=0.3)
    savefig(fig, "fig11_meanlag_bias")
    R["meanlag"] = {f"{t:+.2f}": {k: float(rows[k][i]) for k in rows} for i, t in enumerate(lags)}
    # cost of the nuisance parameter on the null limit
    ul_plain = []; ul_joint = []
    for k in range(30):
        cat = simulate_catalog(SimConfig(n_events=n_events, seed=4500 + k))
        ul_plain.append(upper_limit(*ul_from_cat(cat)[:2])); ul_joint.append(upper_limit(*ul_from_cat(cat, nuisance=("meanlag",))[:2]))
    R["meanlag_UL_cost_pct"] = float((np.median(ul_joint) / np.median(ul_plain) - 1) * 100)


def heavy_tails(n_sim=200, n_events=400):
    out = {}
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for df, lab, col in ((np.inf, "Gaussian lags", "C0"), (3.0, r"Student-$t$ ($\nu=3$) lags", "C3")):
        pulls_g, pulls_r, cov_g, cov_r, pulls_s = [], [], [], [], []
        for k in range(n_sim):
            cat = simulate_catalog(SimConfig(n_events=n_events, lag_df=df, seed=5000 + k))
            th, sg, chi2r = ul_from_cat(cat)
            pulls_s.append(th / (sg * np.sqrt(chi2r)))
            bR, cR, w = robust_fit(cat)
            thr, sgr = bR[0], np.sqrt(cR[0, 0])
            pulls_g.append(th / sg); pulls_r.append(thr / sgr)
            cov_g.append(upper_limit(th, sg) >= 0); cov_r.append(upper_limit(thr, sgr) >= 0)
        pulls_g, pulls_r, pulls_s = np.array(pulls_g), np.array(pulls_r), np.array(pulls_s)
        key = "gauss" if np.isinf(df) else "t3"
        out[key] = {"gls_pull_std": float(pulls_g.std()), "huber_pull_std": float(pulls_r.std()),
                    "chi2scaled_pull_std": float(pulls_s.std()), "chi2scaled_frac_above_1p645": float(np.mean(pulls_s > 1.645)),
                    "gls_frac_above_1p645": float(np.mean(pulls_g > 1.645)), "huber_frac_above_1p645": float(np.mean(pulls_r > 1.645)),
                    "huber_mean_weight": float(np.mean(w))}
        ax[0].hist(pulls_g, bins=25, histtype="step", color=col, label=f"{lab}: std={pulls_g.std():.2f}")
        ax[1].hist(pulls_r, bins=25, histtype="step", color=col, label=f"{lab}, Huber: std={pulls_r.std():.2f}")
        ax[1].hist(pulls_s, bins=25, histtype="step", color=col, ls="--", label=rf"{lab}, $\chi^2$-rescaled: std={pulls_s.std():.2f}")
    xx = np.linspace(-4, 4, 100)
    for a, ttl in zip(ax, ("(a) GLS pull under null", "(b) Huber and $\\chi^2$-rescaled pulls")):
        a.plot(xx, norm.pdf(xx) * n_sim * (8 / 25), color="gray", lw=0.8)
        a.set(xlabel=r"$\hat\theta/\sigma_{\hat\theta}$", ylabel="catalogs", title=ttl); a.legend()
    savefig(fig, "fig13_heavy_tails")
    R["heavy_tails"] = out


def z_errors_and_anchoring(n_sim=50, n_events=400):
    zerrs = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3)
    med_z, bias_z = [], []
    for ze in zerrs:
        uls, ths = [], []
        for k in range(n_sim):
            cat = simulate_catalog(SimConfig(n_events=n_events, z_err_frac=ze, seed=6000 + k + int(ze * 1000)))
            th, sg, _ = ul_from_cat(cat); uls.append(upper_limit(th, sg) * 1e3); ths.append(th / sg)
        med_z.append(np.median(uls)); bias_z.append(np.mean(ths))
    rec_z = []
    for ze in zerrs:
        rec = [np.sqrt(max(ul_from_cat(simulate_catalog(SimConfig(n_events=n_events, z_err_frac=ze, E_t_inject_meV=0.8, seed=6200 + k)))[0], 0)) * 1e3 for k in range(n_sim)]
        rec_z.append(np.mean(rec))
    R["z_err_recovered_0p8"] = {f"{z:.2f}": float(r) for z, r in zip(zerrs, rec_z)}
    fracs = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
    med_f = []
    for f in fracs:
        uls = [upper_limit(*ul_from_cat(simulate_catalog(SimConfig(n_events=n_events, f_anchored=f, seed=6500 + k)))[:2]) * 1e3 for k in range(n_sim)]
        med_f.append(np.median(uls))
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ax[0].plot(zerrs, med_z, "o-", label="median 95% UL")
    ax[0].set(xlabel=r"redshift error $\sigma_z/(1+z)$", ylabel="95% UL on $E_t$ [meV]", title="(a) redshift uncertainty")
    ax0b = ax[0].twinx(); ax0b.plot(zerrs, rec_z, "s--", color="C3", label=r"recovered $\hat E_t$ (0.8 meV injected)"); ax0b.set_ylabel(r"recovered $\hat E_t$ [meV]", color="C3")
    ax0b.set_ylim(0.6, 1.0)
    ax[0].legend(loc="upper left", fontsize=6.5); ax0b.legend(loc="lower right", fontsize=6.5)
    ax[1].plot(fracs, med_f, "o-")
    ax[1].set(xlabel="fraction of GW-anchored events", ylabel="95% UL on $E_t$ [meV]", title="(b) value of an external clock")
    for a in ax:
        a.grid(alpha=0.3)
    savefig(fig, "fig14_zerr_anchoring")
    R["z_err"] = {f"{z:.2f}": {"UL_meV": float(m), "null_pull": float(b)} for z, m, b in zip(zerrs, med_z, bias_z)}
    R["anchored_frac"] = {f"{f:.1f}": float(m) for f, m in zip(fracs, med_f)}


def liv_confusion(n_sim=30, n_events=400):
    """Can a sub-luminal linear LIV delay masquerade as (or hide) a tachyonic advance?"""
    # LIV coefficient 1/E_QG with E_QG = 1e19 GeV = 1e28 eV
    c1 = 1.0 / 1e30   # E_QG = 1e21 GeV, beyond current linear-LIV limits
    # template correlation on the fiducial catalog
    cat = simulate_catalog(SimConfig(n_events=n_events, seed=42))
    KT, KL = [], []
    for rows, A, t, S in _design(cat, nuisance=("liv1",)):
        KT.append(A[:, 0]); KL.append(A[:, 1])
    KT, KL = np.concatenate(KT), np.concatenate(KL)
    R["liv_template_corr"] = float(np.corrcoef(KT, KL)[0, 1])
    res = {}
    for label, Et, liv in (("null", 0.0, 0.0), ("liv_only", 0.0, c1), ("tach_only", 0.6, 0.0), ("both", 0.6, c1)):
        th_p, th_j, liv_j = [], [], []
        for k in range(n_sim):
            catk = simulate_catalog(SimConfig(n_events=n_events, E_t_inject_meV=Et, liv_linear_coeff=liv, seed=7500 + k))
            th_p.append(ul_from_cat(catk)[0])
            b, cv, _, _ = gls_fit(catk, nuisance=("liv1",))
            th_j.append(b[0]); liv_j.append(b[1])
        res[label] = {"theta_plain_1e6": float(np.mean(th_p) * 1e6), "theta_joint_1e6": float(np.mean(th_j) * 1e6),
                      "liv_joint_over_true": float(np.mean(liv_j) / c1) if liv else float(np.mean(liv_j) * 1e28),
                      "theta_true_1e6": float((Et * 1e-3) ** 2 * 1e6)}
    R["liv"] = res
    # figure: recovered theta for the four cases
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    labels = list(res.keys()); x = np.arange(len(labels))
    ax.bar(x - 0.2, [res[l]["theta_plain_1e6"] for l in labels], 0.4, label="tachyon template only")
    ax.bar(x + 0.2, [res[l]["theta_joint_1e6"] for l in labels], 0.4, label="joint tachyon + linear-LIV fit")
    ax.plot(x, [res[l]["theta_true_1e6"] for l in labels], "k_", ms=18, mew=2, label="truth")
    ax.set_xticks(x); ax.set_xticklabels(["null", "LIV only\n($E_{QG}=10^{21}$ GeV)", "tachyon only\n(0.6 meV)", "both"], fontsize=7)
    ax.set(ylabel=r"$\langle\hat\theta\rangle$ [$10^{-6}$ eV$^2$]", title="tachyon vs. sub-luminal LIV")
    ax.legend(fontsize=6.5); ax.grid(alpha=0.3, axis="y")
    savefig(fig, "fig15_liv_confusion")


def jet_delay(n_sim=30, n_events=400):
    """Anchored events in reality carry a positive, unknown jet-launch delay."""
    out = {}
    for label, nuis in (("plain", ()), ("const_nuisance", ("jet",))):
        ths, uls = [], []
        for k in range(n_sim):
            cat = simulate_catalog(SimConfig(n_events=n_events, jet_delay_mean=1.5, jet_delay_sigma=0.7, seed=8000 + k))
            th, sg, _ = ul_from_cat(cat, nuisance=nuis)
            ths.append(th / sg); uls.append(upper_limit(th, sg) * 1e3)
        out[label] = {"mean_pull": float(np.mean(ths)), "median_UL_meV": float(np.median(uls))}
    R["jet_delay"] = out


def gw170817():
    dt_obs, z = 1.74, 0.0099
    Es = np.logspace(4, 6.5, 100)          # 10 keV .. 3 MeV
    # bound: 0.5 (Et/E)^2 H0^-1 I2(z) <= dt_obs  ->  Et <= E sqrt(2 dt_obs H0 / I2)
    I2 = kernel_I2(z)
    Et_bound = Es * np.sqrt(2 * dt_obs / (H0_INV_S * I2))
    R["gw_bound_500keV_meV"] = float(5e5 * np.sqrt(2 * dt_obs / (H0_INV_S * I2)) * 1e3)
    R["gw_bound_50keV_meV"] = float(5e4 * np.sqrt(2 * dt_obs / (H0_INV_S * I2)) * 1e3)
    R["gw_bound_1MeV_meV"] = float(1e6 * np.sqrt(2 * dt_obs / (H0_INV_S * I2)) * 1e3)
    # speed-of-gravity style bound: use -3e-15 lower end of (v_gw - c)/c interval as conservative |v/c - 1| < 3e-15 at E=0.5 MeV
    R["gw_bound_from_speed_3e15_meV"] = float(5e5 * np.sqrt(2 * 3e-15) * 1e3)
    R["gw_dist_Mpc"] = float(z * 299792.458 / H0_KM_S_MPC)  # cz/H0
    R["tachyon_mass_kg_1meV"] = float(tachyon_mass_kg(1e-3))
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.loglog(Es, Et_bound * 1e3, label=r"$\Delta t_{\rm tach}(E)\leq1.74$ s, $z=0.0099$")
    ax.axvspan(1e4, 1e6, color="gray", alpha=0.15, lw=0, label="Fermi-GBM band")
    ax.axhline(R["fid_UL_meV"], ls="--", color="C3", label=f"400-event simulated sensitivity ({R['fid_UL_meV']:.2f} meV)")
    ax.set(xlabel="assumed photon energy $E$ [eV]", ylabel="single-event bound on $E_t$ [meV]", title="GW170817 / GRB 170817A")
    ax.legend(fontsize=6.5); ax.grid(alpha=0.3, which="both")
    savefig(fig, "fig16_gw170817")


# --------------------------------------------------------------------------- #
def write_numbers_tex(path="paper/numbers.tex"):
    def cmd(name, val):
        return f"\\newcommand{{\\{name}}}{{{val}}}\n"
    L = []
    L.append(cmd("fidUL", f"{R['fid_UL_meV']:.2f}"))
    L.append(cmd("fidULbayes", f"{R['fid_UL_bayes_meV']:.2f}"))
    L.append(cmd("fidTheta", f"{R['fid_theta_eV2']*1e7:.1f}"))
    L.append(cmd("fidSigma", f"{R['fid_sigma_eV2']*1e7:.1f}"))
    L.append(cmd("fidChi", f"{R['fid_chi2_red']:.2f}"))
    L.append(cmd("fidRows", f"{R['fid_rows']}"))
    L.append(cmd("fidAnch", f"{R['fid_n_anchored']}"))
    L.append(cmd("fidPull", f"{R['fid_theta_sig']:.2f}"))
    L.append(cmd("ULonetwenty", f"{R['UL_120_median_meV']:.2f}"))
    L.append(cmd("aucA", f"{R['auc']['0.3']:.2f}")); L.append(cmd("aucB", f"{R['auc']['0.5']:.2f}")); L.append(cmd("aucC", f"{R['auc']['1.0']:.2f}"))
    L.append(cmd("scalingTwentyFive", f"{R['scaling']['25']:.2f}")); L.append(cmd("scalingSixteenHundred", f"{R['scaling']['1600']:.2f}"))
    L.append(cmd("scalingSlope", f"{R['scaling_fitted_slope']:.2f}"))
    L.append(cmd("kernelBiasLow", f"{R['kernel_bias_z0p1']:.0f}")); L.append(cmd("kernelBiasHigh", f"{R['kernel_bias_z3']:.0f}")); L.append(cmd("EtBiasHigh", f"{R['Et_bias_z3']:.0f}"))
    L.append(cmd("dtOneMeV", f"{R['dt_1MeV_1meV_z1_ms']:.0f}"))
    L.append(cmd("linBias", f"{R['linearity_max_rel_bias_pct']:.1f}")); L.append(cmd("linThetaBias", f"{R['linearity_theta_max_rel_bias_pct']:.1f}"))
    L.append(cmd("bootRatio", f"{R['boot_sigma_over_analytic']:.2f}"))
    ib = R["info_budget"]
    L.append(cmd("infoMeVanch", f"{100*ib.get('MeV/anchored',0):.0f}")); L.append(cmd("infoMeVunanch", f"{100*ib.get('MeV/unanchored',0):.0f}"))
    L.append(cmd("infoGeV", f"{abs(ib.get('GeV/unanchored',0))+abs(ib.get('GeV/anchored',0)):.0e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}"))
    cov = R["coverage"]
    L.append(cmd("covMin", f"{100*min(v['freq'] for v in cov.values()):.1f}")); L.append(cmd("covMax", f"{100*max(v['freq'] for v in cov.values()):.1f}"))
    L.append(cmd("covBayesMin", f"{100*min(v['bayes'] for v in cov.values()):.1f}"))
    L.append(cmd("covPullStdMax", f"{max(v['pull_std'] for v in cov.values()):.2f}"))
    ml = R["meanlag"]
    L.append(cmd("mlPlainBias", f"{ml['+0.30']['plain']:.2f}")); L.append(cmd("mlJointBias", f"{ml['+0.30']['joint']:.2f}")); L.append(cmd("mlAnchBias", f"{ml['+0.30']['anchored_only']:.2f}"))
    L.append(cmd("mlCost", f"{R['meanlag_UL_cost_pct']:.0f}"))
    ht = R["heavy_tails"]
    L.append(cmd("htGaussStd", f"{ht['gauss']['gls_pull_std']:.2f}")); L.append(cmd("htTthreeStd", f"{ht['t3']['gls_pull_std']:.2f}")); L.append(cmd("htHuberStd", f"{ht['t3']['huber_pull_std']:.2f}"))
    L.append(cmd("htTthreeFA", f"{100*ht['t3']['gls_frac_above_1p645']:.0f}")); L.append(cmd("htHuberFA", f"{100*ht['t3']['huber_frac_above_1p645']:.0f}"))
    L.append(cmd("htScaledStd", f"{ht['t3']['chi2scaled_pull_std']:.2f}")); L.append(cmd("htScaledFA", f"{100*ht['t3']['chi2scaled_frac_above_1p645']:.0f}"))
    L.append(cmd("scalingFourHundred", f"{R['scaling']['400']:.2f}"))
    ze = R["z_err"]
    L.append(cmd("zeUL", f"{ze['0.30']['UL_meV']:.2f}")); L.append(cmd("zeULzero", f"{ze['0.00']['UL_meV']:.2f}")); L.append(cmd("zePull", f"{ze['0.30']['null_pull']:+.2f}"))
    L.append(cmd("zeRec", f"{R['z_err_recovered_0p8']['0.30']:.2f}")); L.append(cmd("zeRecZero", f"{R['z_err_recovered_0p8']['0.00']:.2f}"))
    L.append(cmd("infoAnchEvents", f"{100*R['info_frac_anchored_events']:.0f}")); L.append(cmd("infoHighZ", f"{100*R['info_frac_z_above_1']:.0f}")); L.append(cmd("fracHighZ", f"{100*R['frac_events_z_above_1']:.0f}"))
    af = R["anchored_frac"]
    L.append(cmd("afZero", f"{af['0.0']:.2f}")); L.append(cmd("afOne", f"{af['1.0']:.2f}"))
    L.append(cmd("livCorr", f"{R['liv_template_corr']:.2f}"))
    lv = R["liv"]
    L.append(cmd("livOnlyPlain", f"{lv['liv_only']['theta_plain_1e6']:.2f}")); L.append(cmd("livOnlyJoint", f"{lv['liv_only']['theta_joint_1e6']:.2f}"))
    L.append(cmd("livBothPlain", f"{lv['both']['theta_plain_1e6']:.2f}")); L.append(cmd("livBothJoint", f"{lv['both']['theta_joint_1e6']:.2f}"))
    L.append(cmd("livBothLIV", f"{lv['both']['liv_joint_over_true']:.2f}"))
    jd = R["jet_delay"]
    L.append(cmd("jdPlainPull", f"{jd['plain']['mean_pull']:+.2f}")); L.append(cmd("jdConstPull", f"{jd['const_nuisance']['mean_pull']:+.2f}"))
    L.append(cmd("jdPlainUL", f"{jd['plain']['median_UL_meV']:.2f}")); L.append(cmd("jdConstUL", f"{jd['const_nuisance']['median_UL_meV']:.2f}"))
    L.append(cmd("gwFive", f"{R['gw_bound_500keV_meV']:.0f}")); L.append(cmd("gwFifty", f"{R['gw_bound_50keV_meV']:.1f}")); L.append(cmd("gwOneMeV", f"{R['gw_bound_1MeV_meV']:.0f}"))
    L.append(cmd("gwSpeed", f"{R['gw_bound_from_speed_3e15_meV']:.0f}"))
    L.append(cmd("massKg", f"{R['tachyon_mass_kg_1meV']:.1e}".replace("e-", r"\times10^{-") + "}"))
    with open(path, "w") as f:
        f.write("% auto-generated by tachyon_search.py -- do not edit\n" + "".join(L))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
    q = args.quick
    log("kinematics figures"); fig_kinematics(); fig_velocity_profile(); fig_kernels(); fig_advance_vs_z(); fig_bands_vs_z()
    log("fiducial catalog"); fiducial_and_profile()
    log("linearity"); fig_linearity(n_cat=10 if q else 100)
    log("classifier"); fig_classifier(n_per_class=30 if q else 150)
    log("scaling"); fig_scaling(n_real=5 if q else 30)
    log("coverage"); coverage(n_sim=30 if q else 300)
    log("mean-lag bias"); mean_lag_bias(n_sim=8 if q else 40)
    log("heavy tails"); heavy_tails(n_sim=30 if q else 200)
    log("z errors / anchoring"); z_errors_and_anchoring(n_sim=6 if q else 30)
    log("LIV confusion"); liv_confusion(n_sim=6 if q else 30)
    log("jet delay"); jet_delay(n_sim=6 if q else 30)
    log("GW170817"); gw170817()
    with open("results.json", "w") as f:
        json.dump(R, f, indent=2)
    os.makedirs("paper", exist_ok=True); write_numbers_tex()
    if os.path.isdir("paper_cas"):
        write_numbers_tex("paper_cas/numbers.tex")
    log("done")




# ============================================================================
# 6. Data export
# ============================================================================

def export_data():
    """Write data/*.csv and data/*.json from results.json and the deterministic model."""
    #!/usr/bin/env python3
    """Export the numerical data behind the paper to CSV/JSON files in data/.

    Deterministic quantities (kernels, advances, fiducial catalog, GW170817 bound)
    are recomputed with the same seeds as run_all.py; Monte Carlo summaries are
    taken from results.json.
    """

    os.makedirs("data", exist_ok=True)
    R = json.load(open("results.json"))

    def write_csv(name, header, rows):
        with open(f"data/{name}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)

    # 1. cosmological kernels and advances (Figs 3, 4, 5)
    z = np.linspace(0, 3, 61)
    I1, I2 = kernel_I1(z), kernel_I2(z)
    write_csv("kernels_I1_I2", ["z", "I1", "I2", "I1_over_I2"],
              [(f"{a:.3f}", f"{b:.6f}", f"{c:.6f}", f"{(b/c if c>0 else np.nan):.4f}") for a, b, c in zip(z, I1, I2)])
    write_csv("advance_vs_z_1MeV_1meV", ["z", "dt_naive_ms", "dt_lo_ms", "dt_exact_ms"],
              [(f"{a:.3f}", f"{arrival_advance_naive(1e6,1e-3,a)*1e3:.4f}", f"{arrival_advance_lo(1e6,1e-3,a)*1e3:.4f}", f"{arrival_advance_exact(1e6,1e-3,a)*1e3:.4f}") for a in z])
    zb = np.linspace(0.01, 3, 60)
    write_csv("advance_by_band_1meV", ["z"] + [f"dt_{b}_s" for b in BANDS],
              [[f"{a:.3f}"] + [f"{arrival_advance_lo(BAND_ENERGY[b],1e-3,a):.6e}" for b in BANDS] for a in zb])
    E = np.logspace(5, 15, 101)
    write_csv("kinematics_vs_energy", ["E_eV"] + [f"v_over_c_minus1_Et{Et:.0e}eV" for Et in (1e3,1e5,1e7)] + [f"dt_z1_s_Et{Et:.0e}eV" for Et in (1e3,1e5,1e7)],
              [[f"{e:.4e}"] + [f"{group_velocity_excess(e,Et):.4e}" for Et in (1e3,1e5,1e7)] + [f"{arrival_advance_lo(e,Et,1.0):.4e}" for Et in (1e3,1e5,1e7)] for e in E])

    # 2. fiducial catalog (400 events, seed 42) and its fit
    cat = simulate_catalog(SimConfig(n_events=400, seed=42))
    write_csv("fiducial_catalog_seed42", ["event", "band", "E_eV", "E_ref_eV", "z_true", "z_meas", "anchored", "t_s", "K_s_per_eV2", "sigma_diag_s2", "sigma_shared_s2"],
              [(int(e), BANDS[int(b)], f"{En:.3e}", ("inf" if np.isinf(Er) else f"{Er:.3e}"), f"{zt:.5f}", f"{zm:.5f}", int(a), f"{t:.6f}", f"{K:.6e}", f"{sd:.6f}", f"{ss:.6f}")
               for e, b, En, Er, zt, zm, a, t, K, sd, ss in zip(cat.event, cat.band, cat.E, cat.E_ref, cat.z_true, cat.z_meas, cat.anchored, cat.t, cat.K, cat.sigma_diag, cat.sigma_shared)])
    beta, cov, chi2, ndof = gls_fit(cat)
    fid = {"theta_hat_eV2": float(beta[0]), "sigma_theta_eV2": float(np.sqrt(cov[0, 0])), "chi2": float(chi2), "ndof": int(ndof),
           "Et_UL95_meV": float(upper_limit(beta[0], np.sqrt(cov[0, 0])) * 1e3), "n_rows": int(len(cat.t)),
           "n_events_usable": int(len(np.unique(cat.event))), "information_budget": {k: float(v) for k, v in information_budget(cat).items()}}
    boot = bootstrap_theta(cat, n_boot=1000)
    write_csv("fiducial_bootstrap_theta", ["theta_hat_eV2"], [(f"{b:.6e}",) for b in boot])
    json.dump(fid, open("data/fiducial_fit.json", "w"), indent=2)

    # 3. GW170817 bound versus assumed photon energy (Fig 16)
    Es = np.logspace(4, 6.5, 100); I2gw = kernel_I2(0.0099)
    write_csv("gw170817_bound_vs_energy", ["E_eV", "Et_bound_meV"], [(f"{e:.4e}", f"{e*np.sqrt(2*1.74/(H0_INV_S*I2gw))*1e3:.4f}") for e in Es])

    # 4. Monte Carlo summaries from results.json
    write_csv("scaling_UL_vs_N", ["N", "median_UL95_meV"], [(k, f"{v:.4f}") for k, v in R["scaling"].items()])
    write_csv("coverage", ["Et_inj_meV", "coverage_freq", "coverage_flatprior", "pull_mean", "pull_std"],
              [(k, f"{v['freq']:.4f}", f"{v['bayes']:.4f}", f"{v['pull_mean']:.4f}", f"{v['pull_std']:.4f}") for k, v in R["coverage"].items()])
    write_csv("meanlag_bias", ["tau_s_per_decade", "theta_plain_1e-6eV2", "theta_joint_1e-6eV2", "theta_anchored_only_1e-6eV2"],
              [(k, f"{v['plain']:.4f}", f"{v['joint']:.4f}", f"{v['anchored_only']:.4f}") for k, v in R["meanlag"].items()])
    write_csv("redshift_error", ["sigma_z_over_1pz", "median_UL95_meV", "null_pull_mean", "recovered_Et_meV_for_0p8_injected"],
              [(k, f"{v['UL_meV']:.4f}", f"{v['null_pull']:.4f}", f"{R['z_err_recovered_0p8'][k]:.4f}") for k, v in R["z_err"].items()])
    write_csv("anchored_fraction", ["f_anchored", "median_UL95_meV"], [(k, f"{v:.4f}") for k, v in R["anchored_frac"].items()])
    write_csv("liv_confusion", ["case", "theta_true_1e-6eV2", "theta_plain_1e-6eV2", "theta_joint_1e-6eV2", "liv_recovered_over_true"],
              [(k, f"{v['theta_true_1e6']:.4f}", f"{v['theta_plain_1e6']:.4f}", f"{v['theta_joint_1e6']:.4f}", f"{v['liv_joint_over_true']:.4f}") for k, v in R["liv"].items()])
    write_csv("heavy_tails", ["lag_distribution", "gls_pull_std", "gls_frac_above_1p645", "chi2scaled_pull_std", "chi2scaled_frac_above_1p645", "huber_pull_std", "huber_frac_above_1p645"],
              [(k, f"{v['gls_pull_std']:.4f}", f"{v['gls_frac_above_1p645']:.4f}", f"{v['chi2scaled_pull_std']:.4f}", f"{v['chi2scaled_frac_above_1p645']:.4f}", f"{v['huber_pull_std']:.4f}", f"{v['huber_frac_above_1p645']:.4f}") for k, v in R["heavy_tails"].items()])
    write_csv("jet_delay", ["fit", "mean_null_pull", "median_UL95_meV"], [(k, f"{v['mean_pull']:.4f}", f"{v['median_UL_meV']:.4f}") for k, v in R["jet_delay"].items()])
    write_csv("classifier_auc", ["Et_inj_meV", "AUC"], [(k, f"{v:.4f}") for k, v in R["auc"].items()])

    # 5. everything else, verbatim
    json.dump(R, open("data/results_all.json", "w"), indent=2)
    with open("data/README.txt", "w") as f:
        f.write("""Data underlying the figures and tables of
    'A population-level matched-filter search for tachyonic arrival-time advances
    in gamma-ray bursts and their gravitational-wave counterparts' (V. Singh).

    Units: energies in eV, times in s, theta = E_t^2 in eV^2 unless a column name says otherwise.

    kinematics_vs_energy.csv        Fig. 1  velocity excess and z=1 advance vs observed energy
    kernels_I1_I2.csv               Fig. 3  cosmological kernels and their ratio
    advance_vs_z_1MeV_1meV.csv      Fig. 4  naive / leading-order / exact advance
    advance_by_band_1meV.csv        Fig. 5  advance per observing band
    fiducial_catalog_seed42.csv     Sec. 5.1 simulated fiducial catalog (one row per event-band)
    fiducial_fit.json               Sec. 5.1 GLS fit of that catalog, information budget
    fiducial_bootstrap_theta.csv    Fig. 8a 1000 bootstrap resamples of theta_hat
    classifier_auc.csv              Fig. 9  cross-validated AUC per injected scale
    scaling_UL_vs_N.csv             Fig. 10 median 95% limit vs catalog size
    coverage.csv                    Fig. 11 coverage and pulls vs injected E_t
    meanlag_bias.csv                Fig. 12 bias from a systematic spectral lag
    jet_delay.csv                   Sec. 6.4 anchored-event jet delay test
    heavy_tails.csv                 Fig. 13 Student-t lag test
    redshift_error.csv              Fig. 14a redshift-error test
    anchored_fraction.csv           Fig. 14b anchored-fraction sweep
    liv_confusion.csv               Fig. 15 tachyon / linear-LIV joint fit
    gw170817_bound_vs_energy.csv    Fig. 16 single-event bound vs assumed photon energy
    results_all.json                every number in the paper, machine-readable

    Fiducial simulation settings: 400 events, sigma_lag = 0.5 s (Gaussian), 20% GW-anchored,
    log-normal redshifts (median 0.8, log-width 0.7, 0.005 <= z <= 5), Planck 2018 flat LCDM.
    Band energies / timing / detection probabilities: MeV 0.5 MeV 10 ms 1.00; GeV 1 GeV 100 ms 0.45;
    TeV 1 TeV 0.5 s 0.15; nu 100 TeV 1 s 0.08.
    """)
    print("wrote", len(os.listdir("data")), "files to data/")



if __name__ == "__main__":
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--quick", action="store_true", help="reduced Monte Carlo")
    _ap.add_argument("--export", action="store_true", help="also write data/*.csv")
    _args = _ap.parse_args()
    import sys
    sys.argv = [sys.argv[0]] + (["--quick"] if _args.quick else [])
    main()
    if _args.export:
        export_data()