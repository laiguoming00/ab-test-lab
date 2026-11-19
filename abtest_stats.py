"""Statistics helpers for the A/B test classroom demo (no Streamlit imports)."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ProportionResult:
    n_a: int
    n_b: int
    conv_a: int
    conv_b: int
    rate_a: float
    rate_b: float
    abs_diff: float
    rel_lift: float
    ci_low: float  # 95% CI on absolute difference (B - A)
    ci_high: float
    z: float
    p_value: float


def two_proportion_test(conv_a: int, n_a: int, conv_b: int, n_b: int) -> ProportionResult | None:
    """Two-sided pooled z-test for B vs A, plus unpooled 95% CI for the difference."""
    if n_a == 0 or n_b == 0:
        return None
    pa, pb = conv_a / n_a, conv_b / n_b
    pool = (conv_a + conv_b) / (n_a + n_b)
    se_pool = sqrt(pool * (1 - pool) * (1 / n_a + 1 / n_b))
    z = (pb - pa) / se_pool if se_pool > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    se = sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)
    d = pb - pa
    return ProportionResult(
        n_a, n_b, conv_a, conv_b, pa, pb, d,
        d / pa if pa > 0 else float("nan"),
        d - 1.96 * se, d + 1.96 * se, z, p,
    )


def welch_test(a: pd.Series, b: pd.Series) -> tuple[float, float] | None:
    """Welch t-test on (e.g.) revenue per visitor. Returns (t, p)."""
    if len(a) < 2 or len(b) < 2 or (a.std() == 0 and b.std() == 0):
        return None
    t, p = stats.ttest_ind(b, a, equal_var=False)
    return float(t), float(p)


def srm_check(n_a: int, n_b: int) -> float | None:
    """Sample-ratio-mismatch p-value for an intended 50/50 split."""
    n = n_a + n_b
    if n == 0:
        return None
    return float(stats.chisquare([n_a, n_b], [n / 2, n / 2]).pvalue)


def sample_size_per_arm(baseline: float, rel_mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Visitors needed per arm to detect a relative lift of rel_mde over baseline."""
    p1 = baseline
    p2 = min(baseline * (1 + rel_mde), 0.9999)
    pbar = (p1 + p2) / 2
    za = stats.norm.ppf(1 - alpha / 2)
    zb = stats.norm.ppf(power)
    num = (za * sqrt(2 * pbar * (1 - pbar)) + zb * sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p2 - p1) ** 2))


def simulate_peeking(p: float, n_per_look: int, looks: int, sims: int, seed: int | None = None):
    """Run `sims` A/A tests (no real difference) and check significance after every look.

    Returns (share significant at the final look only,
             share that were significant at ANY look,
             |z| paths for the first 15 experiments as a DataFrame).
    """
    rng = np.random.default_rng(seed)
    a = rng.binomial(n_per_look, p, size=(sims, looks)).cumsum(axis=1)
    b = rng.binomial(n_per_look, p, size=(sims, looks)).cumsum(axis=1)
    n = n_per_look * np.arange(1, looks + 1)
    pool = (a + b) / (2 * n)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(2 * pool * (1 - pool) / n)
        z = np.where(se > 0, (b / n - a / n) / se, 0.0)
    sig = np.abs(z) > 1.96
    paths = pd.DataFrame(np.abs(z[:15]).T, index=pd.Index(np.arange(1, looks + 1), name="look"))
    paths.columns = [f"exp {i + 1}" for i in range(paths.shape[1])]
    return float(sig[:, -1].mean()), float(sig.any(axis=1).mean()), paths
