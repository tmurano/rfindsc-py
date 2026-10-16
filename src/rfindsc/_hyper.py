"""Log upper tail of the hypergeometric distribution (internal).

``log_upper_tail(M, G, K, P)`` returns ``log P(X >= M)`` for
``X ~ Hypergeometric(K successes, P - K failures, G draws)``, vectorised over
numpy arrays. This is the quantity R computes with
``phyper(M - 1, K, P - K, G, lower.tail = FALSE, log.p = TRUE)``, and the same
numerical strategy is used:

* the point probability is evaluated on the log scale with Loader's
  saddle-point expansion (C. Loader, "Fast and accurate computation of
  binomial probabilities", 2000): a hypergeometric density is a ratio of three
  binomial densities, each written with Stirling-series errors and deviance
  terms, so it keeps full relative accuracy (no cancellation of large
  log-factorials);
* the tail is the point probability times the sum of successive probability
  ratios, summed on the side of the distribution away from the mode until the
  terms drop below machine precision; the opposite tail is ``log(1 - exp(.))``.

Everything stays on the log scale, so tails far below 1e-308 are exact.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

_LN_2PI = math.log(2.0 * math.pi)
_LN2 = math.log(2.0)
_EPS = float(np.finfo(np.float64).eps)

# stirlerr(n) = log(n!) - (n + 1/2) log(n) + n - log(2 pi) / 2 for n = 0..15,
# computed with 50-digit arithmetic (n = 0 is never used).
_STIRLERR_SMALL = np.array([
    0.0,
    0.08106146679532726,
    0.0413406959554093,
    0.02767792568499834,
    0.020790672103765093,
    0.016644691189821193,
    0.013876128823070748,
    0.01189670994589177,
    0.010411265261972096,
    0.009255462182712733,
    0.00833056343336287,
    0.007573675487951841,
    0.00694284010720953,
    0.006408994188004207,
    0.0059513701127588475,
    0.005554733551962801,
])

# Stirling series stirlerr(n) = 1/(12 n) - 1/(360 n^3) + 1/(1260 n^5) - ...
# (B_2k / (2k (2k - 1)) n^(1 - 2k)); eight terms are exact to double precision
# for n >= 16.
_STIRLING_SERIES = (1 / 12, -1 / 360, 1 / 1260, -1 / 1680, 1 / 1188,
                    -691 / 360360, 1 / 156, -3617 / 122400)


@lru_cache(maxsize=16)
def _stirlerr_table(nmax: int) -> np.ndarray:
    """stirlerr(n) for the integers n = 0..nmax."""
    out = np.empty(nmax + 1)
    k = min(nmax, 15)
    out[: k + 1] = _STIRLERR_SMALL[: k + 1]
    if nmax > 15:
        n = np.arange(16, nmax + 1, dtype=np.float64)
        inv2 = 1.0 / (n * n)
        acc = np.full_like(n, _STIRLING_SERIES[-1])
        for c in _STIRLING_SERIES[-2::-1]:
            acc = c + acc * inv2
        out[16:] = acc / n
    out.setflags(write=False)
    return out


def _bd0(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Deviance term ``x log(x / mu) + mu - x`` (x > 0, mu > 0).

    Near ``x == mu`` the direct formula cancels, so a series in
    ``v = (x - mu) / (x + mu)`` is summed instead.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = x * np.log(x / mu) + mu - x
    near = np.abs(x - mu) < 0.1 * (x + mu)
    if near.any():
        xn, mn = x[near], mu[near]
        v = (xn - mn) / (xn + mn)
        s = (xn - mn) * v
        ej = 2.0 * xn * v
        v2 = v * v
        s1 = s
        for j in range(1, 1000):
            ej = ej * v2
            s1 = s + ej / (2 * j + 1)
            if np.array_equal(s1, s):
                break
            s = s1
        out[near] = s1
    return out


def _log_dbinom(k: np.ndarray, n: np.ndarray, p: np.ndarray, q: np.ndarray,
                st: np.ndarray) -> np.ndarray:
    """log dbinom(k; n, p) with q = 1 - p passed separately (Loader 2000)."""
    out = np.full(k.shape, -np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        q0 = q == 0
        out[q0 & (k == n)] = 0.0
        ok = ~q0 & (k >= 0) & (k <= n)
        z = ok & (k == 0)
        if z.any():
            nz, pz, qz = n[z], p[z], q[z]
            out[z] = np.where(nz == 0, 0.0,
                              np.where(pz > qz, nz * np.log(qz), nz * np.log1p(-pz)))
        f = ok & (k == n) & (k > 0)
        if f.any():
            nf, pf, qf = n[f], p[f], q[f]
            out[f] = np.where(pf > qf, nf * np.log1p(-qf), nf * np.log(pf))
        i = np.flatnonzero(ok & (k > 0) & (k < n))
        if i.size:
            ki, ni = k[i], n[i]
            kf, nf = ki.astype(np.float64), ni.astype(np.float64)
            lc = (st[ni] - st[ki] - st[ni - ki]
                  - _bd0(kf, nf * p[i]) - _bd0(nf - kf, nf * q[i]))
            lf = _LN_2PI + np.log(kf) + np.log1p(-kf / nf)
            out[i] = lc - 0.5 * lf
    return out


def _log_dbinom_at_mean(n, N, st):
    """log dbinom(n; N, n / N): at the mean both deviance terms are 0."""
    out = np.zeros(n.shape)
    i = np.flatnonzero((n > 0) & (n < N))
    if i.size:
        ni, Ni = n[i], N[i]
        nf = ni.astype(np.float64)
        out[i] = (st[Ni] - st[ni] - st[Ni - ni]) - 0.5 * (_LN_2PI + np.log(nf) + np.log1p(-nf / Ni))
    return out


def _log_dhyper(x, NR, NB, n, st):
    """log P(X = x) for X ~ Hypergeometric(NR successes, NB failures, n draws)."""
    N = NR + NB
    Nf = N.astype(np.float64)
    p = n / Nf
    q = (N - n) / Nf
    return (_log_dbinom(x, NR, p, q, st) + _log_dbinom(n - x, NB, p, q, st)
            - _log_dbinom_at_mean(n, N, st))


def _lower_ratio_sum(x, NR, NB, n):
    """Sum over j >= 1 of P(X = x - j) / P(X = x).

    Terms are added until one falls below machine precision relative to the
    running sum (the density is log-concave, so later terms are smaller still);
    a term becomes exactly 0 at the end of the support. Convergence is checked
    every few steps, so up to three extra terms, each below that precision,
    may be added.
    """
    total = np.zeros(x.shape)
    idx = np.flatnonzero(x > 0)
    xa = x[idx].astype(np.float64)
    c1 = (NB[idx] - n[idx]).astype(np.float64)       # NB - n
    c2 = n[idx].astype(np.float64) + 1                # n + 1
    c3 = NR[idx].astype(np.float64) + 1               # NR + 1
    t = np.ones(idx.size)
    s = np.zeros(idx.size)
    while idx.size:
        for _ in range(4):
            t *= xa * (c1 + xa) / (c2 - xa) / (c3 - xa)
            s += t
            xa -= 1
        go = (t > 0) & (t >= _EPS * s)
        if not go.all():
            done = ~go
            total[idx[done]] = s[done]
            idx, xa, c1, c2, c3, t, s = (a[go] for a in (idx, xa, c1, c2, c3, t, s))
    return total


def _log1mexp(v):
    """log(1 - exp(v)) for v <= 0, accurate over the whole range."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > -_LN2, np.log(-np.expm1(v)), np.log1p(-np.exp(v)))


def log_upper_tail(M, G, K, P: int) -> np.ndarray:
    """log P(X >= M) for X ~ Hypergeometric(K successes, P - K failures, G draws).

    ``M``, ``G`` and ``K`` are integer arrays of the same shape (valid
    parameters: ``0 <= K <= P``, ``0 <= G <= P``); ``P`` is a scalar.
    """
    M = np.asarray(M, dtype=np.int64)
    G = np.asarray(G, dtype=np.int64)
    K = np.asarray(K, dtype=np.int64)
    P = int(P)
    st = _stirlerr_table(P)

    x = M - 1                               # P(X >= M) = P(X > x)
    swap = x * P > G * K                    # x above the mean: sum the other tail
    NR = np.where(swap, P - K, K)
    NB = P - NR
    x = np.where(swap, G - x - 1, x)
    lower = swap                            # after swapping, the lower tail is wanted

    out = np.empty(x.shape)
    t0 = (x < 0) | (x < G - NB)             # tail probability 0 (lower) / 1 (upper)
    t1 = ~t0 & ((x >= NR) | (x >= G))       # tail probability 1 (lower) / 0 (upper)
    out[t0] = np.where(lower[t0], -np.inf, 0.0)
    out[t1] = np.where(lower[t1], 0.0, -np.inf)
    i = np.flatnonzero(~(t0 | t1))
    if i.size:
        xi, NRi, NBi, ni, lw = x[i], NR[i], NB[i], G[i], lower[i]
        d = _log_dhyper(xi, NRi, NBi, ni, st)
        res = np.where(lw, -np.inf, 0.0)    # point probability 0: as at t0
        f = np.flatnonzero(d > -np.inf)
        if f.size:
            v = d[f] + np.log1p(_lower_ratio_sum(xi[f], NRi[f], NBi[f], ni[f]))
            res[f] = np.where(lw[f], v, _log1mexp(v))
        out[i] = res
    return out
