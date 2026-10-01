"""Data for the first-survivor experiment. No plotting, colours or layout.

first_survivor_data() computes every even N in the main range, the worked target,
the dyadic envelope and its fits, and the complete prime-square blocks inside the
range. exact_blocks() computes the selected blocks beyond it. Every number the
figures print comes from these objects.
"""
from dataclasses import dataclass

import numpy as np

from goldbach import number_theory as nt


@dataclass(frozen=True)
class Sweep:
    """Every even N with 6 <= N < n_max, as parallel arrays."""
    N: np.ndarray               # the even targets
    q: np.ndarray               # largest prime below √N
    W: np.ndarray               # forcing boundary q² - N/2
    lam_sieve: np.ndarray       # first-survivor offset λ_sieve(N)
    lam_prime: np.ndarray       # nearest Goldbach-pair offset λ_prime(N) (OEIS A047160 at N/2)
    first_prime: np.ndarray     # bool: the first surviving pair is a pair of primes
    first_composite: np.ndarray  # least offset of a surviving pair that is not a prime pair, or -1
    n_composite: np.ndarray     # number of such offsets


@dataclass(frozen=True)
class WorkedExample:
    """The worked target. Offsets are 0 <= d <= C - 2."""
    N: int
    C: int
    q: int
    W: int
    lam_sieve: int
    lam_prime: int
    survivors: np.ndarray       # every surviving offset
    composite: np.ndarray       # surviving offsets whose pair is not a pair of primes
    n_inside: int               # surviving offsets d < W
    n_outside: int              # surviving offsets d >= W
    inside_all_prime: bool      # every surviving pair with d < W is a pair of primes
    parity_class: int           # d mod 2 of the offsets not removed by 2 (C + 1 mod 2)
    centre_d: np.ndarray        # offsets of that class in the centre zoom
    centre_alive: np.ndarray    # which of them survive
    edge_d: np.ndarray          # offsets of that class in the boundary zoom
    edge_alive: np.ndarray      # which of them survive
    edge_composite: np.ndarray  # which of them are composite survivors


@dataclass(frozen=True)
class EnvelopeFit:
    """Fits to the dyadic-bin maxima inside N < 2^K, each bin maximum placed at the
    least N attaining it.

    alpha_N is the fitted exponent: OLS of log λ on log N. The figure's q-unit value
    2·alpha_N is a conversion via N ≈ q², approximate because q < √N; alpha_q_direct
    is a separate OLS of log λ on log q at the same points, kept for comparison only.
    beta is the exponent of the alternative λ ≈ B·(log N)^β, fitted the same way.
    """
    cutoff_exp: int             # K
    n_bins: int
    alpha_N: float
    log_A: float                # λ ≈ exp(log_A)·N^alpha_N
    rss_power: float            # residual sum of squares of the power law, in log λ
    alpha_q_direct: float
    beta: float
    log_B: float                # λ ≈ exp(log_B)·(log N)^beta
    rss_log: float
    x_range: tuple              # (first, last) N of the fitted maxima


@dataclass(frozen=True)
class Block:
    """One complete prime-square block q² < N < r², every even N computed. Every N in
    it has sieve depth q, so W = q² - N/2 falls linearly across it."""
    q: int
    r: int
    n_targets: int
    max_lam: int                # largest λ_sieve in the block
    argN: int                   # the least N attaining it
    min_W: int                  # smallest forcing boundary in the block
    max_ratio: float            # largest λ_sieve / W in the block
    all_inside: bool            # λ_sieve < W for every N in the block


@dataclass(frozen=True)
class OffsetHistogram:
    """How the λ_sieve values are distributed, column by column of log N.

    share[i, j] is the fraction of the even N in column j whose λ_sieve lies in row i.
    Rows are single integers 0..LINEAR_ROWS - 1, then log-spaced, so λ = 0 keeps its row.
    """
    x_edges: np.ndarray
    y_edges: np.ndarray
    share: np.ndarray
    col_max: np.ndarray         # largest λ_sieve in each column


@dataclass(frozen=True)
class FirstSurvivorData:
    sweep: Sweep
    example: WorkedExample
    n_max: int                  # N < n_max
    exceptions: np.ndarray      # N with λ_sieve >= W
    composite_first: np.ndarray  # of those, N whose first surviving pair is not a prime pair
    prime_outside: np.ndarray   # of those, N whose first surviving pair is a prime pair
    min_N: int                  # first N after the last exception: λ_sieve < W for min_N <= N < n_max
    n_with_composite: int       # N >= min_N with at least one composite survivor
    max_ratio: float            # largest λ_sieve / W for N >= min_N
    max_ratio_N: int
    bin_exps: np.ndarray        # bins [2^k, 2^(k+1)) by k
    bin_max: np.ndarray         # largest λ_sieve in each bin
    bin_argN: np.ndarray        # the least N attaining it
    bin_min_W: np.ndarray       # smallest W in each bin
    fits: tuple                 # EnvelopeFit per cutoff, in the order given
    blocks: tuple               # complete prime-square blocks inside the range, from min_N
    histogram: OffsetHistogram  # λ_sieve for N >= min_N


def dyadic_maxima(N, values, k_lo, k_hi):
    """(ks, maxima, argN) over the bins [2^k, 2^(k+1)), k_lo <= k < k_hi. argN is the
    least N in the bin attaining the maximum. N must be sorted."""
    ks, mx, arg = [], [], []
    for k in range(k_lo, k_hi):
        i, j = np.searchsorted(N, [2 ** k, 2 ** (k + 1)])
        if j <= i:
            raise ValueError(f"bin 2^{k} has no data")
        t = int(np.argmax(values[i:j]))
        ks.append(k); mx.append(int(values[i + t])); arg.append(int(N[i + t]))
    return np.array(ks), np.array(mx), np.array(arg)


def line_fit(x, y):
    """(slope, intercept, residual sum of squares) of the least-squares line."""
    (slope, icpt), res = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1, full=True)[:2]
    return float(slope), float(icpt), float(res[0]) if len(res) else 0.0


def envelope_fit(K, N, lam, q):
    """EnvelopeFit over the points with N < 2^K."""
    s = N < 2 ** K
    x, y = np.log(N[s].astype(float)), np.log(lam[s].astype(float))
    aN, logA, rss_p = line_fit(x, y)
    aq, _, _ = line_fit(np.log(q[s].astype(float)), y)
    beta, logB, rss_l = line_fit(np.log(x), y)
    return EnvelopeFit(K, int(s.sum()), aN, logA, rss_p, aq, beta, logB, rss_l,
                       (int(N[s][0]), int(N[s][-1])))


def block_summary(q, r, Ns, lam, W):
    u = int(np.argmax(lam))
    return Block(int(q), int(r), len(Ns), int(lam[u]), int(Ns[u]), int(W.min()),
                 float((lam / W).max()), bool((lam < W).all()))


LINEAR_ROWS = 10                # histogram rows 0..9 are one integer each


def offset_histogram(N, lam, n_cols=96, n_log_rows=40):
    hi = int(lam.max()) + 1
    y_edges = np.concatenate([np.arange(LINEAR_ROWS + 1) - 0.5,
                              np.geomspace(LINEAR_ROWS + 0.5, hi * 1.05, n_log_rows + 1)[1:]])
    x_edges = np.geomspace(N.min(), N.max() + 2, n_cols + 1)
    counts, _, _ = np.histogram2d(lam, N, bins=[y_edges, x_edges])
    share = counts / counts.sum(axis=0, keepdims=True)
    col = np.clip(np.searchsorted(x_edges, N, side="right") - 1, 0, n_cols - 1)
    col_max = np.zeros(n_cols, dtype=np.int64)
    np.maximum.at(col_max, col, lam)
    return OffsetHistogram(x_edges, y_edges, share, col_max)


def worked_example(N, spf, primes, lam_sieve, lam_prime, centre_zoom, edge_zoom):
    C = N // 2
    q = int(nt.largest_prime_below_sqrt([N], primes)[0])
    W = q * q - C
    d = np.arange(0, C - 1, dtype=np.int64)
    alive = (spf[C - d] >= q) & (spf[C + d] >= q)
    surv = d[alive]
    prime_pair = (spf[C - surv] == C - surv) & (spf[C + surv] == C + surv)
    comp = surv[~prime_pair]
    par = (C + 1) % 2

    def zoom(lo, hi):
        z = np.arange(lo + (lo - par) % 2, hi + 1, 2, dtype=np.int64)
        return z, (spf[C - z] >= q) & (spf[C + z] >= q)

    cd, ca = zoom(0, centre_zoom)
    ed, ea = zoom(W - edge_zoom, min(W + edge_zoom, C - 2))
    return WorkedExample(N, C, q, W, lam_sieve, lam_prime, surv, comp,
                         int((surv < W).sum()), int((surv >= W).sum()), bool(prime_pair[surv < W].all()),
                         par, cd, ca, ed, ea, np.isin(ed, comp))


def first_survivor_data(example_N, n_max, bin_start_exp=8, fit_cutoff_exps=(14, 17, 20),
                        centre_zoom=120, edge_zoom=90):
    """Every even N with 6 <= N < n_max, n_max a power of 2 so that the dyadic bins
    [2^k, 2^(k+1)) from k = bin_start_exp are complete. Each fit uses the bins inside
    N < 2^K for K in fit_cutoff_exps. The worked example must be inside the range."""
    kmax = n_max.bit_length() - 1
    if n_max != 2 ** kmax or max(fit_cutoff_exps) > kmax:
        raise ValueError("n_max must be a power of 2 covering every fit cutoff")
    if example_N % 2 or not 6 <= example_N < n_max:
        raise ValueError("example N must be even and inside the range")
    spf = nt.spf_table(n_max)
    primes = nt.primes_from_spf(spf)
    N = np.arange(6, n_max, 2, dtype=np.int64)
    q = nt.largest_prime_below_sqrt(N, primes)
    W = nt.forcing_boundary(N, q)
    ls = nt.first_survivor_offsets(N, q, spf)
    lp = nt.nearest_goldbach_offsets(N, spf)
    if (ls < 0).any() or (lp < 0).any():
        raise ValueError("an offset search found nothing")
    C = N // 2
    first_prime = (spf[C - ls] == C - ls) & (spf[C + ls] == C + ls)
    fc, nc = nt.composite_survivors(N, q, spf, primes)
    sweep = Sweep(N, q, W, ls, lp, first_prime, fc, nc)

    exc = ls >= W
    min_N = int(N[exc].max()) + 2 if exc.any() else 6
    if 2 ** bin_start_exp <= min_N:
        raise ValueError("the dyadic bins must start above the exceptions")
    big = N >= min_N
    ratio = ls[big] / W[big]
    t = int(np.argmax(ratio))

    ks, bmax, barg = dyadic_maxima(N, ls, bin_start_exp, kmax)
    bmin_W = np.array([W[(N >= 2 ** k) & (N < 2 ** (k + 1))].min() for k in ks])
    qa = q[np.searchsorted(N, barg)]
    fits = tuple(envelope_fit(K, barg, bmax, qa) for K in fit_cutoff_exps)

    blocks = []
    for qq in np.unique(q[big]).tolist():
        r = int(primes[np.searchsorted(primes, qq) + 1])
        if r * r >= n_max:
            break
        i, j = np.searchsorted(q, [qq, qq + 1])
        blocks.append(block_summary(qq, r, N[i:j], ls[i:j], W[i:j]))

    i = int(np.searchsorted(N, example_N))
    ex = worked_example(example_N, spf, primes, int(ls[i]), int(lp[i]), centre_zoom, edge_zoom)
    return FirstSurvivorData(sweep, ex, n_max, N[exc], N[exc & ~first_prime], N[exc & first_prime],
                             min_N, int((nc[big] > 0).sum()), float(ratio[t]), int(N[big][t]),
                             ks, bmax, barg, bmin_W, fits, tuple(blocks), offset_histogram(N[big], ls[big]))


# ---- exact blocks beyond the main range

def selected_block_qs(targets, primes):
    """The largest prime at or below each target, without repeats, in order."""
    out = []
    for t in targets:
        p = int(primes[np.searchsorted(primes, t, side="right") - 1])
        if p not in out:
            out.append(p)
    return out


def exact_blocks(targets):
    """The complete prime-square block of q = the largest prime at or below each
    target. Every even N in each block is computed; nothing is sampled."""
    # Each q <= t = max(targets), and by Bertrand's postulate there is a prime in
    # (t, 2t), so primes up to 2t include the next prime r after every q.
    primes = nt.primes_from_spf(nt.spf_table(2 * int(max(targets))))
    blocks = []
    for q in selected_block_qs(targets, primes):
        r = int(primes[np.searchsorted(primes, q) + 1])
        Ns, lam = nt.block_first_survivor_offsets(q, r, primes)
        blocks.append(block_summary(q, r, Ns, lam, q * q - Ns // 2))
    return tuple(blocks)
