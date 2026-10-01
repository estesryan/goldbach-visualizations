"""Mathematical tests and brute-force cross-checks for the first-survivor experiment.

Primitives in goldbach/number_theory.py are checked by hand and against
tests/first_survivor_brute_force.py, which never imports the code under test.
The experiment's data (first_survivor fixture, conftest.py) is then recomputed
independently: λ_sieve from the gcd definition, both λ from a pure-Python
least-prime-factor list over the whole range, the exceptions, composite survivors,
the worked example, the bins and fits, and the exact blocks. Headline numbers are
asserted literally because the README and the report quote them.
"""
from math import gcd, log

import numpy as np
import pytest

import brute_force as bf
import first_survivor_brute_force as fbf
from extras.first_survivor_experiment import data as fd
from extras.first_survivor_experiment import generate
from goldbach import number_theory as nt

pytestmark = pytest.mark.first_survivor


@pytest.fixture(scope="module")
def spf_2_20():
    """Pure-Python least prime factors of every n <= 2^20."""
    return bf.smallest_prime_factors(2 ** 20)


# ---- primitives

def test_spf_table_and_primes():
    spf = nt.spf_table(10_000)
    assert spf.tolist() == bf.smallest_prime_factors(10_000)
    assert nt.primes_from_spf(spf).tolist() == nt.primes_between(2, 10_001)


def test_largest_prime_below_sqrt():
    primes = nt.primes_from_spf(nt.spf_table(2000))
    Ns = list(range(6, 40_000, 2))
    assert nt.largest_prime_below_sqrt(Ns, primes).tolist() == [fbf.sieve_depth(N) for N in Ns]
    assert nt.largest_prime_below_sqrt([999_008, 1_000_000, 2 ** 20 - 2], primes).tolist() == [997, 997, 1021]
    for bad in ([4], [7], [101]):
        with pytest.raises(ValueError):
            nt.largest_prime_below_sqrt(bad, primes)


def test_forcing_boundary_known_values():
    assert nt.forcing_boundary([100, 272, 999_008], [7, 13, 997]).tolist() == [-1, 33, 494_505]


def test_first_survivor_and_nearest_goldbach_offset_by_hand():
    # N = 8: no prime lies below q = 2, so d = 0, the pair (4, 4), survives; the nearest prime pair is 3 + 5.
    # N = 100: 47 + 53 survives the primes below 7. N = 272: 109 + 163 at d = 27, below W = 33.
    spf = nt.spf_table(1000)
    primes = nt.primes_from_spf(spf)
    Ns = np.array([8, 16, 44, 100, 272])
    qs = nt.largest_prime_below_sqrt(Ns, primes)
    assert nt.first_survivor_offsets(Ns, qs, spf).tolist() == [0, 1, 3, 3, 27]
    assert nt.nearest_goldbach_offsets(Ns, spf).tolist() == [1, 3, 9, 3, 27]


def test_nearest_goldbach_offsets_match_oeis_a047160():
    # A047160(n) for n = 3..22, from the OEIS entry, at N = 2n.
    spf = nt.spf_table(100)
    assert nt.nearest_goldbach_offsets(2 * np.arange(3, 23), spf).tolist() == [
        0, 1, 0, 1, 0, 3, 2, 3, 0, 1, 0, 3, 2, 3, 0, 1, 0, 3, 2, 9]


def test_offset_search_stops_at_c_minus_2():
    # A boundary-condition test: N = 12 has sieve depth 3, but q = 7 is supplied on
    # purpose so that no valid offset survives. (6, 6), (5, 7), (4, 8), (3, 9) and
    # (2, 10) each have a factor below 7; the next offset would be (1, 11), outside
    # 0 <= d <= C - 2, so the search must report -1 rather than continue.
    spf = nt.spf_table(20)
    assert nt.first_survivor_offsets([12], [7], spf).tolist() == [-1]
    assert nt.nearest_goldbach_offsets([12], spf).tolist() == [1]


def test_composite_survivors_by_hand():
    # N = 272: (103, 169 = 13²) survives the primes below 13 at d = 33 = W, and is the only one.
    # N = 999,008: (4,999, 994,009 = 997²) at d = 494,505 = W, also the only one.
    spf = nt.spf_table(1_000_000)
    primes = nt.primes_from_spf(spf)
    Ns = np.array([272, 999_008])
    first, count = nt.composite_survivors(Ns, nt.largest_prime_below_sqrt(Ns, primes), spf, primes)
    assert first.tolist() == [33, 494_505] and count.tolist() == [1, 1]


def test_rough_segment_matches_least_prime_factors():
    spf = nt.spf_table(20_000)
    primes = nt.primes_from_spf(spf)
    assert nt.rough_segment(10_000, 20_000, 41, primes).tolist() == (spf[10_000:20_001] >= 41).tolist()
    with pytest.raises(ValueError):
        nt.rough_segment(30, 100, 41, primes)


def test_block_offsets_match_the_main_computation():
    spf = nt.spf_table(2 ** 16)
    primes = nt.primes_from_spf(spf)
    Ns, lam = nt.block_first_survivor_offsets(211, 223, primes, margin=8)   # small margin: exercises doubling
    assert Ns[0] == 211 ** 2 + 1 and Ns[-1] == 223 ** 2 - 1
    assert lam.tolist() == nt.first_survivor_offsets(Ns, np.full(len(Ns), 211), spf).tolist()


# ---- the sweep against brute force

def test_sweep_covers_every_even_n(first_survivor):
    sw = first_survivor.sweep
    assert sw.N.tolist() == list(range(6, 2 ** 20, 2))
    for a in (sw.q, sw.W, sw.lam_sieve, sw.lam_prime, sw.first_prime, sw.first_composite, sw.n_composite):
        assert len(a) == len(sw.N)


def test_sieve_depth_over_the_range(first_survivor, spf_2_20):
    sw = first_survivor.sweep
    assert all(fbf.sieve_depth_spf(N, spf_2_20) == q for N, q in zip(sw.N.tolist(), sw.q.tolist()))
    # composite_survivors relies on q³ >= N: a composite with no factor below q is q·p'.
    assert (sw.q ** 3 >= sw.N).all()


def test_lambda_sieve_matches_the_gcd_definition(first_survivor):
    sw = first_survivor.sweep
    s = sw.N < 4000
    for N, lam in zip(sw.N[s].tolist(), sw.lam_sieve[s].tolist()):
        assert fbf.first_survivor_gcd(N) == lam, N


def test_both_lambdas_over_the_whole_range(first_survivor, spf_2_20):
    sw = first_survivor.sweep
    ls, lp = sw.lam_sieve.tolist(), sw.lam_prime.tolist()
    for i, N in enumerate(sw.N.tolist()):
        assert fbf.both_lambdas_spf(N, spf_2_20) == (ls[i], lp[i]), N


def test_lambdas_by_trial_division_on_a_sample(first_survivor):
    rng = np.random.default_rng(20_261_001)
    sw = first_survivor.sweep
    for i in rng.choice(len(sw.N), 300, replace=False).tolist():
        N = int(sw.N[i])
        assert fbf.nearest_goldbach_offset_trial(N) == sw.lam_prime[i], N
        C, q, lam = N // 2, fbf.sieve_depth(N), int(sw.lam_sieve[i])
        assert fbf.survives_gcd(C, lam, q)
        assert not any(fbf.survives_gcd(C, d, q) for d in range(lam)), N


def test_exceptions_by_brute_force(first_survivor):
    exc, comp = [], []
    for N in range(6, 4000, 2):
        C, q = N // 2, fbf.sieve_depth(N)
        lam = fbf.first_survivor_gcd(N)
        if lam >= q * q - C:
            exc.append(N)
            if not (bf.is_prime(C - lam) and bf.is_prime(C + lam)):
                comp.append(N)
    d = first_survivor
    assert exc == d.exceptions.tolist()
    assert comp == d.composite_first.tolist()
    assert len(exc) == 22 and len(comp) == 13
    assert exc == [8, 16, 18, 20, 22, 24, 44, 48, 92, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116,
                   118, 120]
    assert comp == [8, 16, 18, 20, 44, 48, 92, 96, 98, 102, 108, 110, 116]
    assert d.prime_outside.tolist() == [22, 24, 100, 104, 106, 112, 114, 118, 120]


def test_headline_range(first_survivor):
    # The last exception is N = 120, the end of the q = 7 block, so λ_sieve < W from
    # N = 122 on; the earlier statement "from N = 132" holds a fortiori.
    d = first_survivor
    sw = d.sweep
    inside = sw.lam_sieve < sw.W
    assert d.min_N == 122
    assert inside[sw.N >= 122].all() and inside[sw.N >= 132].all() and not inside[sw.N == 120].all()
    assert sw.q[sw.N == 120][0] == 7 and sw.q[sw.N == 122][0] == 11
    assert (sw.W[sw.N >= 122] > 0).all() and (sw.W[sw.N <= 120] <= 0).sum() > 0


def test_inside_w_the_first_survivor_is_the_nearest_prime_pair(first_survivor):
    sw = first_survivor.sweep
    inside = sw.lam_sieve < sw.W
    assert (sw.lam_sieve[inside] == sw.lam_prime[inside]).all()
    assert sw.first_prime[inside].all()
    # Composite first survivors are exactly where λ_sieve < λ_prime.
    assert sw.N[sw.lam_sieve != sw.lam_prime].tolist() == first_survivor.composite_first.tolist()


def test_largest_ratio_above_the_exceptions(first_survivor):
    d = first_survivor
    sw = d.sweep
    best = max((ls / W, N) for N, ls, W in zip(sw.N.tolist(), sw.lam_sieve.tolist(), sw.W.tolist()) if N >= 122)
    assert (d.max_ratio, d.max_ratio_N) == best
    assert d.max_ratio == pytest.approx(27 / 33) and d.max_ratio_N == 272


def test_composite_survivors_by_enumeration(first_survivor, spf_2_20):
    # Every surviving offset enumerated directly, for all N below 4,000 and 12 sampled N
    # above: no survivor below W fails to be a prime pair, and the first and count match.
    sw = first_survivor.sweep
    rng = np.random.default_rng(7)
    idx = np.nonzero(sw.N < 4000)[0].tolist() + rng.choice(np.nonzero(sw.N >= 4000)[0], 12, replace=False).tolist()
    for i in idx:
        N = int(sw.N[i])
        bad = [d for d, both_prime in fbf.surviving_offsets_spf(N, spf_2_20) if not both_prime]
        assert all(d >= sw.W[i] for d in bad), N
        assert sw.n_composite[i] == len(bad), N
        assert sw.first_composite[i] == (min(bad) if bad else -1), N


def test_no_composite_survivor_below_w(first_survivor):
    d = first_survivor
    sw = d.sweep
    has = sw.n_composite > 0
    assert (sw.first_composite[has] >= sw.W[has]).all()
    assert (sw.first_composite[~has] == -1).all()
    assert d.n_with_composite == int(has[sw.N >= d.min_N].sum()) == 175_468


def test_boundary_composite_where_n_minus_q_squared_survives(first_survivor, spf_2_20):
    # The pair at d = W is (N − q², q²). From N = 122, W >= 0, so d = W is a valid offset
    # (W <= C - 2) exactly when N − q² >= 2; it fails at N = q² + 1. When valid, q²
    # survives, so W holds a composite survivor exactly when N − q² has no prime factor
    # below q, and it is then the first composite survivor.
    sw = first_survivor.sweep
    for N, q, W, fc in zip(sw.N.tolist(), sw.q.tolist(), sw.W.tolist(), sw.first_composite.tolist()):
        if N >= 122:
            m = N - q * q
            assert W >= 0 and (W <= N // 2 - 2) == (m >= 2), N
            assert (fc == W) == (m >= 2 and spf_2_20[m] >= q), N


def test_worked_example(first_survivor, spf_2_20):
    e = first_survivor.example
    C, q = 999_008 // 2, fbf.sieve_depth(999_008)
    assert (e.N, e.C, e.q, e.W) == (999_008, C, q, q * q - C) == (999_008, 499_504, 997, 494_505)
    assert not any(fbf.survives_gcd(C, d, q) for d in range(45)) and fbf.survives_gcd(C, 45, q)
    assert fbf.nearest_goldbach_offset_trial(999_008) == 45
    assert (e.lam_sieve, e.lam_prime) == (45, 45)
    assert bf.is_prime(C - 45) and bf.is_prime(C + 45) and (C - 45, C + 45) == (499_459, 499_549)
    offs = fbf.surviving_offsets_spf(999_008, spf_2_20)
    assert e.survivors.tolist() == [d for d, _ in offs]
    assert (len(offs), e.n_inside, e.n_outside) == (4_007, 3_956, 51)
    assert e.inside_all_prime and all(p for d, p in offs if d < e.W)
    assert [d for d, p in offs if not p] == e.composite.tolist() == [494_505]
    assert (C - 494_505, C + 494_505) == (4_999, 997 ** 2) and bf.is_prime(4_999)
    # the zooms: the odd offsets (C is even), the first survivor in the centre, the composite at W
    assert e.parity_class == 1 and (e.centre_d % 2 == 1).all() and (e.edge_d % 2 == 1).all()
    assert e.centre_d[e.centre_alive].tolist() == [d for d, _ in offs if d <= generate.CENTRE_ZOOM]
    assert e.edge_d[e.edge_composite].tolist() == [494_505]


def test_boundary_segments_and_block_transitions(first_survivor):
    # Within a block W drops by exactly 1 per step of 2 in N (slope −1/2); each block
    # starts at the first even N above q², for consecutive primes q.
    sw = first_survivor.sweep
    same = sw.q[1:] == sw.q[:-1]
    assert (np.diff(sw.W)[same] == -1).all()
    starts = np.nonzero(np.r_[True, ~same])[0]
    qs = sw.q[starts].tolist()
    assert qs == bf.primes_between(2, max(qs) + 1)
    assert sw.N[starts].tolist() == [max(6, q * q + 1 + (q * q + 1) % 2) for q in qs]


# ---- the envelope and fits

def test_dyadic_maxima_by_plain_loop(first_survivor):
    d = first_survivor
    sw = d.sweep
    assert d.bin_exps.tolist() == list(range(8, 20))
    for k, mx, arg, mw in zip(d.bin_exps.tolist(), d.bin_max.tolist(), d.bin_argN.tolist(), d.bin_min_W.tolist()):
        best, where, low = -1, None, None
        for N, lam, W in zip(sw.N.tolist(), sw.lam_sieve.tolist(), sw.W.tolist()):
            if 2 ** k <= N < 2 ** (k + 1):
                if lam > best:
                    best, where = lam, N
                low = W if low is None else min(low, W)
        assert (best, where, low) == (mx, arg, mw), k
    assert d.bin_max.tolist() == [39, 75, 93, 168, 228, 300, 369, 525, 621, 810, 1086, 1281]


def test_fits_by_normal_equations(first_survivor):
    d = first_survivor
    for f in d.fits:
        s = d.bin_exps + 1 <= f.cutoff_exp
        xs, ys = d.bin_argN[s].tolist(), d.bin_max[s].tolist()
        lx, ly = [log(x) for x in xs], [log(y) for y in ys]
        b, a, rss = fbf.ols(lx, ly)
        assert (f.alpha_N, f.log_A, f.rss_power) == pytest.approx((b, a, rss), abs=1e-10)
        bq, _, _ = fbf.ols([log(fbf.sieve_depth(x)) for x in xs], ly)
        assert f.alpha_q_direct == pytest.approx(bq, abs=1e-10)
        bl, al, rl = fbf.ols([log(v) for v in lx], ly)
        assert (f.beta, f.log_B, f.rss_log) == pytest.approx((bl, al, rl), abs=1e-10)
        assert f.n_bins == len(xs) and f.x_range == (xs[0], xs[-1])


def test_fit_values(first_survivor):
    # The README quotes these.
    fits = first_survivor.fits
    assert [f.cutoff_exp for f in fits] == [14, 17, 20]
    assert [f.n_bins for f in fits] == [6, 9, 12]
    assert [round(f.alpha_N, 3) for f in fits] == [0.603, 0.513, 0.445]
    assert [round(f.alpha_q_direct, 3) for f in fits] == [1.099, 0.986, 0.867]
    assert [round(f.beta, 2) for f in fits] == [4.58, 4.37, 4.23]
    assert [round(f.rss_power, 3) for f in fits] == [0.116, 0.212, 0.421]
    assert [round(f.rss_log, 3) for f in fits] == [0.114, 0.123, 0.153]
    # The fitted exponent drifts down as the range grows, and the polylog model fits at
    # least as well over every range: the figure's growth statement depends on this.
    assert fits[0].alpha_N > fits[1].alpha_N > fits[2].alpha_N
    assert all(f.rss_log <= f.rss_power for f in fits)


def test_q_unit_conversion_is_not_the_direct_q_fit(first_survivor):
    # 2α_N is a conversion via N ≈ q²; the direct fit against q is a different number
    # because q < √N. The figure shows only the conversion.
    for f in first_survivor.fits:
        assert abs(2 * f.alpha_N - f.alpha_q_direct) > 0.02


def test_fit_start_sensitivity():
    # README: starting the bins at 2^11 instead of 2^8 changes the exponents.
    d = fd.first_survivor_data(generate.EXAMPLE_N, generate.N_MAX, 11, generate.FIT_CUTOFF_EXPS)
    assert [round(f.alpha_N, 3) for f in d.fits] == [0.534, 0.415, 0.371]


def test_histogram_accounts_for_every_n(first_survivor):
    d = first_survivor
    h = d.histogram
    big = d.sweep.N >= d.min_N
    assert np.allclose(h.share.sum(axis=0), 1)
    counts, _, _ = np.histogram2d(d.sweep.lam_sieve[big], d.sweep.N[big], bins=[h.y_edges, h.x_edges])
    assert counts.sum() == big.sum()
    assert np.allclose(h.share, counts / counts.sum(axis=0))
    assert h.y_edges[0] == -0.5 and h.y_edges[1] == 0.5          # λ = 0 has its own row
    assert counts[0].sum() == (d.sweep.lam_sieve[big] == 0).sum() > 0


def test_main_range_blocks(first_survivor):
    d = first_survivor
    sw = d.sweep
    assert d.blocks[0].q == 11 and d.blocks[-1].r ** 2 < 2 ** 20
    assert [b.q for b in d.blocks] == bf.primes_between(11, d.blocks[-1].q + 1)
    for b in d.blocks:
        s = (sw.N > b.q ** 2) & (sw.N < b.r ** 2)
        assert b.n_targets == s.sum() == (b.r ** 2 - b.q ** 2) // 2
        assert b.max_lam == sw.lam_sieve[s].max() and b.argN == sw.N[s][np.argmax(sw.lam_sieve[s])]
        assert b.min_W == sw.W[s].min() and b.all_inside


def test_data_rejects_bad_settings():
    with pytest.raises(ValueError):
        fd.first_survivor_data(1000, 1_000_000)                     # not a power of 2
    with pytest.raises(ValueError):
        fd.first_survivor_data(999, 2 ** 12, fit_cutoff_exps=(10,))
    with pytest.raises(ValueError):
        fd.first_survivor_data(1000, 2 ** 12, fit_cutoff_exps=(14,))
    with pytest.raises(ValueError):
        fd.first_survivor_data(1000, 2 ** 12, bin_start_exp=6, fit_cutoff_exps=(10,))   # bins reach the exceptions


# ---- exact blocks beyond the main range

def test_selected_blocks(first_survivor_blocks):
    blocks = first_survivor_blocks
    assert [b.q for b in blocks] == [max(bf.primes_between(2, t + 1)) for t in generate.BLOCK_TARGETS]
    assert len(blocks) == 16 and blocks[0].q == 1_097 and blocks[-1].q == 49_999
    assert all(b.q ** 2 >= 2 ** 20 for b in blocks)                 # beyond the main range
    for b in blocks:
        assert b.r == min(p for p in range(b.q + 1, 2 * b.q) if bf.is_prime(p))
        assert b.q ** 2 < b.argN < b.r ** 2 and b.argN % 2 == 0
        assert b.n_targets == (b.r ** 2 - b.q ** 2) // 2
        assert b.all_inside and b.max_ratio < 1
    assert sum(b.n_targets for b in blocks) == 3_395_436


def test_selected_block_maxima_by_the_gcd_definition(first_survivor_blocks):
    # At each block's maximising N: every smaller offset is dead by the gcd definition,
    # the maximum survives, and its pair is prime by trial division.
    for b in first_survivor_blocks:
        C = b.argN // 2
        P = fbf.primorial_below(b.q)
        assert all(gcd(C * C - d * d, P) > 1 for d in range(b.max_lam)), b.q
        assert gcd(C * C - b.max_lam ** 2, P) == 1
        assert bf.is_prime(C - b.max_lam) and bf.is_prime(C + b.max_lam)


def test_first_selected_block_exhaustively(first_survivor_blocks):
    # Block q = 1,097: every N recomputed from a pure-Python least-prime-factor list,
    # independent of the segmented sieve.
    b = first_survivor_blocks[0]
    spf = bf.smallest_prime_factors(b.r ** 2)
    best, where, ratio = -1, None, 0.0
    for N in range(b.q ** 2 + 1, b.r ** 2, 2):
        C, d = N // 2, 0
        while not (spf[C - d] >= b.q and spf[C + d] >= b.q):
            d += 1
        W = b.q * b.q - C
        assert d < W
        ratio = max(ratio, d / W)
        if d > best:
            best, where = d, N
    assert (b.max_lam, b.argN) == (best, where) and b.max_ratio == pytest.approx(ratio)
